"""
物理计算层模块

实现粒子属性随时间演化的物理法则和空间投影：
- TimePhysics: 物理演化函数
- ParticleProjector: 欧式空间到庞加莱球的投影
"""
import math
import torch
import torch.nn.functional as F


class TimePhysics:
    """
    物理演化层：计算粒子属性随时间的演化
    
    目前 f 和 g 均为恒等映射，为未来记忆固化与遗忘曲线预留接口。
    """
    @staticmethod
    def f(v: float, t_born: float, t_now: float) -> float:
        """
        速度/强度演化函数 f(v, t)
        
        Args:
            v: 初始速度/强度
            t_born: 粒子产生时间
            t_now: 当前计算时间
            
        Returns:
            当前时刻的速度/强度
        """
        # 默认实现: f(x, t) = x (速度恒定，无阻力)
        # 边界处理: 速度不能为负
        return max(0.0, v)

    @staticmethod
    def g(T: float, t_born: float, t_now: float) -> float:
        """
        温度演化函数 g(T, t)
        
        Args:
            T: 初始温度
            t_born: 粒子产生时间
            t_now: 当前计算时间
            
        Returns:
            当前时刻的温度
        """
        # 默认实现: g(x, t) = x (温度恒定，无冷却)
        # 边界处理: 温度不能为负
        return max(0.0, T)


class ParticleProjector:
    """
    空间投影层：欧式空间 -> 庞加莱球
    
    将物理属性映射为双曲几何坐标，支持任意曲率的双曲空间。
    """
    def __init__(self, curvature: float = 1.0, scaling_factor: float = 2.0):
        """
        Args:
            curvature (c): 双曲空间的曲率，默认 1.0。曲率越大，空间弯曲程度越高。
            scaling_factor: 强度映射放大系数，默认 2.0。系数越大，同等强度下离圆心越远。
        """
        self.c = curvature
        self.scaling_factor = scaling_factor
        # 预计算 sqrt(c) 避免重复计算
        self.sqrt_c = math.sqrt(curvature)

    def compute_state(self, 
                      vec: torch.Tensor, 
                      v: float, 
                      T: float, 
                      born: float, 
                      t_now: float) -> dict:
        """
        计算粒子在当前时刻的动态状态（双曲坐标、当前速度、当前温度）
        
        性能优化：直接接收原始数值，避免构建 Point 对象的开销。
        
        Args:
            vec: 情感向量 (torch.Tensor)
            v: 初始速度/强度
            T: 初始温度
            born: 生成时间戳
            t_now: 当前时间戳
            
        Returns:
            包含以下键的字典:
            - current_vector: 庞加莱球坐标 (torch.Tensor)
            - current_v: 当前速度/强度 (float)
            - current_T: 当前温度 (float)
        """
        # 1. 物理演化：计算当前时刻的 v 和 T
        v_current = TimePhysics.f(v, born, t_now)
        T_current = TimePhysics.g(T, born, t_now)
        
        # 2. 空间投影：欧式 -> 双曲
        # 零向量保护：避免归一化零向量导致的 NaN
        vec_norm = torch.norm(vec)
        if vec_norm < 1e-9:
            direction = torch.zeros_like(vec)
        else:
            direction = F.normalize(vec, p=2, dim=-1)
        
        # 广义双曲投影公式
        # r = tanh(sqrt(c) * dist / 2) / sqrt(c)
        # 这里 dist = v_current * scaling_factor
        arg = self.sqrt_c * v_current * self.scaling_factor / 2.0
        r = torch.tanh(torch.tensor(arg, dtype=vec.dtype)) / self.sqrt_c
        
        poincare_coord = r * direction
        
        return {
            "current_vector": poincare_coord,
            "current_v": v_current,
            "current_T": T_current
        }

