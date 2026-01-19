#!/usr/bin/env python
"""
实验完成后发送邮件通知

读取实验结果并发送关键信息到指定邮箱
"""
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

# 邮件配置
SMTP_SERVER = "smtp.qq.com"
SMTP_PORT = 587
SENDER_EMAIL = "1587105806@qq.com"  # 发送邮箱
SENDER_PASSWORD = ""  # QQ邮箱授权码（需要在QQ邮箱设置中获取，不是QQ密码）
RECIPIENT_EMAIL = "1587105806@qq.com"  # 接收邮箱

def load_experiment_results(results_dir):
    """加载实验结果"""
    results_dir = Path(results_dir)
    
    comparison_file = results_dir / "comparison_results.json"
    baseline_file = results_dir / "baseline_results.json"
    enhanced_file = results_dir / "enhanced_results.json"
    
    results = {}
    
    if comparison_file.exists():
        with open(comparison_file, 'r', encoding='utf-8') as f:
            results['comparison'] = json.load(f)
    
    if baseline_file.exists():
        with open(baseline_file, 'r', encoding='utf-8') as f:
            results['baseline'] = json.load(f)
    
    if enhanced_file.exists():
        with open(enhanced_file, 'r', encoding='utf-8') as f:
            results['enhanced'] = json.load(f)
    
    return results

def format_results_email(results):
    """格式化实验结果为邮件内容"""
    email_content = f"""
HyperAmy 严谨对比实验完成报告

实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

========================================
实验结果摘要
========================================

"""
    
    if 'comparison' in results:
        comp = results['comparison']
        email_content += f"数据集: {comp.get('dataset', 'HotpotQA')}\n"
        email_content += f"查询数量: {comp.get('num_queries', 'N/A')}\n"
        email_content += f"情感权重: {comp.get('sentiment_weight', 'N/A')}\n\n"
        
        email_content += "Baseline 结果:\n"
        for metric, value in sorted(comp.get('baseline', {}).items()):
            email_content += f"  {metric}: {value:.4f}\n"
        
        email_content += "\n情感增强版本结果:\n"
        for metric, value in sorted(comp.get('enhanced', {}).items()):
            email_content += f"  {metric}: {value:.4f}\n"
        
        email_content += "\n改进幅度:\n"
        for metric, improvement in comp.get('improvements', {}).items():
            abs_val = improvement.get('absolute', 0)
            rel_val = improvement.get('relative', 0)
            email_content += f"  {metric}: {abs_val:+.4f} ({rel_val:+.2f}%)\n"
    
    email_content += f"""

========================================
详细结果
========================================

结果文件位置: {Path(results_dir).absolute()}

主要文件:
- comparison_results.json: 对比结果
- baseline_results.json: Baseline详细结果
- enhanced_results.json: 情感增强详细结果

========================================
实验说明
========================================

这是一个严谨的对比实验，使用标准数据集（HotpotQA）和标准评估指标（Recall@K, MRR）进行对比。

实验对比了：
1. Baseline: 标准 HippoRAG（无情感增强）
2. Enhanced: HippoRAGEnhanced（情感权重 0.3）

所有结果已保存在 outputs/rigorous_experiment/ 目录下。

---
HyperAmy 项目
"""
    
    return email_content

def send_email(subject, body, recipient):
    """发送邮件"""
    if not SENDER_PASSWORD:
        print("⚠️  邮件发送功能未配置（需要设置SENDER_PASSWORD）")
        print("邮件内容预览:")
        print("=" * 50)
        print(body)
        print("=" * 50)
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ 邮件已发送到 {recipient}")
        return True
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        print("邮件内容预览:")
        print("=" * 50)
        print(body)
        print("=" * 50)
        return False

if __name__ == "__main__":
    import sys
    
    # 获取结果目录
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # 默认路径
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / "outputs" / "rigorous_experiment"
    
    print("=" * 70)
    print("实验完成邮件通知")
    print("=" * 70)
    
    # 加载结果
    print(f"\n【1】加载实验结果...")
    results = load_experiment_results(results_dir)
    
    if not results:
        print(f"❌ 未找到实验结果文件在: {results_dir}")
        sys.exit(1)
    
    print(f"✅ 结果已加载")
    
    # 格式化邮件内容
    print(f"\n【2】格式化邮件内容...")
    email_body = format_results_email(results)
    
    # 发送邮件
    print(f"\n【3】发送邮件...")
    subject = f"HyperAmy 严谨对比实验完成 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    send_email(subject, email_body, RECIPIENT_EMAIL)
    
    print("\n" + "=" * 70)
    print("✅ 完成")
    print("=" * 70)

