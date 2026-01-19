#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HyperAmy索引后自动验证脚本
监控索引完成，然后自动运行验证实验
"""
import sys
import os
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
log_file = project_root / 'auto_validate_after_indexing.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 服务器配置
# 检测是否在远程服务器上运行（通过检查路径是否存在）
RUN_LOCAL = Path("/public/jiangh/HyperAmy").exists()
if RUN_LOCAL:
    # 在远程服务器上运行
    PROJECT_ROOT = "/public/jiangh/HyperAmy"
    SERVER = None
else:
    # 在本地运行，需要通过SSH
    SERVER = "hyperamy-server"
    PROJECT_ROOT = "/public/jiangh/HyperAmy"

INDEX_LOG = f"{PROJECT_ROOT}/test_hyperamy_parallel_rebuild.log"
INDEX_PID_FILE = f"{PROJECT_ROOT}/test_hyperamy_parallel.pid"
DB_PATH = f"{PROJECT_ROOT}/outputs/three_methods_comparison_monte_cristo/hyperamy_db"
MAPPING_FILE = f"{PROJECT_ROOT}/outputs/three_methods_comparison_monte_cristo/hyperamy_id_to_content.json"

# 验证脚本路径
VALIDATION_SCRIPT = f"{PROJECT_ROOT}/test/test_hyperamy_quick_validation.py"

def check_indexing_process(server=None):
    """检查索引进程是否还在运行"""
    if RUN_LOCAL:
        cmd = "ps aux | grep '[p]ython.*test_hyperamy_parallel.py' | grep python"
    else:
        cmd = f'ssh {server or SERVER} "ps aux | grep \'[p]ython.*test_hyperamy_parallel.py\' | grep python"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            return {
                'running': True,
                'pid': parts[1],
                'time': parts[9]
            }
    except:
        pass
    return {'running': False}

def check_indexing_completion(server=None):
    """检查索引是否完成"""
    # 方法1: 检查日志中的完成信息
    if RUN_LOCAL:
        cmd = f"cd {PROJECT_ROOT} && tail -50 {INDEX_LOG} 2>/dev/null | grep -E '✅.*索引完成|存储了.*个点' | tail -1"
    else:
        cmd = f'ssh {server or SERVER} "cd {PROJECT_ROOT} && tail -50 {INDEX_LOG} 2>/dev/null | grep -E \'✅.*索引完成|存储了.*个点\' | tail -1"'
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            completion_line = result.stdout.strip()
            if '✅' in completion_line or '存储了' in completion_line:
                logger.info(f"检测到索引完成信息: {completion_line}")
                return {'completed': True, 'method': 'log'}
    except:
        pass
    
    # 方法2: 检查数据库和映射文件
    if RUN_LOCAL:
        # 直接在本地运行Python代码
        try:
            db_path = Path(DB_PATH)
            mapping_file = Path(MAPPING_FILE)
            
            if mapping_file.exists():
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    id_to_content = json.load(f)
                mapping_count = len(id_to_content)
                
                if db_path.exists():
                    try:
                        from poincare.storage import HyperAmyStorage
                        storage = HyperAmyStorage(persist_path=str(db_path))
                        db_count = storage.collection.count()
                        logger.info(f"数据库点数: {db_count}, 映射文件点数: {mapping_count}")
                        if db_count >= 9000 and mapping_count >= 9000:
                            logger.info("检测到数据库和映射文件已就绪")
                            return {'completed': True, 'method': 'database'}
                    except Exception as e:
                        logger.warning(f"读取数据库时出错: {e}")
        except Exception as e:
            logger.warning(f"检查数据库时出错: {e}")
    
    return {'completed': False}

def run_validation_test(server=None):
    """运行验证测试（10个查询）"""
    logger.info("=" * 80)
    logger.info("开始运行验证测试（10个查询）...")
    logger.info("=" * 80)
    
    # 运行验证脚本
    if RUN_LOCAL:
        # 使用bash来运行，确保source命令可用
        cmd = f"cd {PROJECT_ROOT} && bash -c 'source /opt/conda/etc/profile.d/conda.sh && conda activate PyTorch-2.4.1 && timeout 300 python -u {VALIDATION_SCRIPT}' 2>&1 | tee test_hyperamy_quick_validation_auto.log"
    else:
        cmd = f'ssh {server or SERVER} "cd {PROJECT_ROOT} && bash -c \\\"source /opt/conda/etc/profile.d/conda.sh && conda activate PyTorch-2.4.1 && timeout 300 python -u {VALIDATION_SCRIPT}\\\" 2>&1 | tee test_hyperamy_quick_validation_auto.log"'
    
    try:
        logger.info("执行验证脚本...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=360)  # 6分钟超时
        
        logger.info("=" * 80)
        logger.info("验证测试执行完成")
        logger.info("=" * 80)
        
        # 检查结果
        if result.returncode == 0:
            # 检查日志中是否有成功信息
            if '验证成功' in result.stdout or 'Recall@' in result.stdout:
                logger.info("✅ 验证测试成功完成")
                return {'success': True, 'output': result.stdout}
            else:
                logger.warning("⚠️  验证测试完成，但未找到明确的成功信息")
                logger.info("输出前500字符:")
                logger.info(result.stdout[:500])
                return {'success': False, 'output': result.stdout}
        else:
            logger.error(f"❌ 验证测试执行失败，返回码: {result.returncode}")
            logger.error("错误输出:")
            logger.error(result.stderr[:500] if result.stderr else "无错误输出")
            return {'success': False, 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        logger.error("❌ 验证测试超时（超过6分钟）")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        logger.error(f"❌ 运行验证测试时出错: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """主函数：监控索引完成，然后自动运行验证"""
    logger.info("=" * 80)
    logger.info("HyperAmy索引后自动验证脚本")
    logger.info("=" * 80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"运行模式: {'本地运行' if RUN_LOCAL else 'SSH远程运行'}")
    if not RUN_LOCAL:
        logger.info(f"服务器: {SERVER}")
    logger.info(f"索引日志: {INDEX_LOG}")
    logger.info(f"数据库路径: {DB_PATH}")
    logger.info("=" * 80)
    
    # 检查索引进程是否存在
    process_info = check_indexing_process()
    if process_info['running']:
        logger.info(f"✅ 检测到索引进程正在运行 (PID: {process_info['pid']}, 运行时间: {process_info['time']})")
        logger.info("等待索引完成...")
    else:
        logger.info("⚠️  未检测到索引进程，可能已完成或未启动")
    
    # 轮询检查索引是否完成
    max_wait_time = 3600 * 2  # 最多等待2小时
    check_interval = 30  # 每30秒检查一次
    waited_time = 0
    max_checks = max_wait_time // check_interval
    
    logger.info(f"开始监控索引完成状态（每{check_interval}秒检查一次，最多等待{max_wait_time//60}分钟）...")
    
    for check_num in range(max_checks):
        # 检查索引是否完成
        completion_info = check_indexing_completion()
        
        if completion_info['completed']:
            logger.info("=" * 80)
            logger.info("✅ 索引已完成！")
            logger.info(f"检测方法: {completion_info['method']}")
            logger.info("=" * 80)
            
            # 等待几秒确保数据库写入完成
            logger.info("等待5秒确保数据库写入完成...")
            time.sleep(5)
            
            # 运行验证测试
            validation_result = run_validation_test()
            
            # 输出总结
            logger.info("=" * 80)
            logger.info("自动验证总结")
            logger.info("=" * 80)
            logger.info(f"索引状态: ✅ 已完成")
            logger.info(f"验证测试: {'✅ 成功' if validation_result.get('success') else '❌ 失败'}")
            
            if validation_result.get('success'):
                logger.info("")
                logger.info("🎉 索引和验证测试均成功完成！")
                logger.info("下一步可以运行完整实验（50个查询）")
            else:
                logger.info("")
                logger.warning("⚠️  验证测试失败，请检查日志")
                if validation_result.get('error'):
                    logger.warning(f"错误信息: {validation_result.get('error')}")
            
            logger.info("=" * 80)
            return validation_result.get('success', False)
        
        # 检查进程是否还在运行
        process_info = check_indexing_process()
        if not process_info['running'] and waited_time > 60:
            # 进程已停止，但索引可能还在完成中，等待一段时间后再检查
            logger.info(f"索引进程已停止，等待60秒后再次检查完成状态...")
            time.sleep(60)
            completion_info = check_indexing_completion()
            if completion_info['completed']:
                logger.info("✅ 索引已完成！")
                validation_result = run_validation_test()
                return validation_result.get('success', False)
            else:
                logger.warning("⚠️  索引进程已停止，但未检测到完成状态，可能已失败")
                logger.warning("请手动检查索引日志")
                return False
        
        waited_time += check_interval
        if check_num % 4 == 0:  # 每2分钟打印一次状态
            logger.info(f"等待中... 已等待 {waited_time//60} 分钟 ({waited_time}/{max_wait_time} 秒)")
        
        time.sleep(check_interval)
    
    logger.error("=" * 80)
    logger.error("❌ 超时：索引未在预期时间内完成")
    logger.error("请手动检查索引状态")
    logger.error("=" * 80)
    return False

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n用户中断，退出监控")
        sys.exit(1)
    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

