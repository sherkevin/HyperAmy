"""
物理计算层模块

实现粒子属性随时间演化的物理法则和空间投影：
- TimePhysics: 物理演化函数
- ParticleProjector: 欧式空间到庞加莱球的投影
"""
import math
import numpy as np
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
    粒子距离原点随时间增加，增加量为速度与时间的积分。
    """
    def __init__(self, curvature: float = 1.0, scaling_factor: float = 2.0, max_radius: float = 100.0):
        """
        Args:
            curvature (c): 双曲空间的曲率，默认 1.0。曲率越大，空间弯曲程度越高。
            scaling_factor: 强度映射放大系数，默认 2.0。系数越大，同等强度下离圆心越远。
            max_radius: 庞加莱球的最大半径，默认 100.0。超过此半径的粒子被认为已消失。
        """
        self.c = curvature
        self.scaling_factor = scaling_factor
        self.max_radius = max_radius
        # 预计算 sqrt(c) 避免重复计算
        self.sqrt_c = math.sqrt(curvature)

    def compute_state(self, 
                      vec, 
                      v: float, 
                      T: float, 
                      born: float, 
                      t_now: float,
                      weight: float = 1.0) -> dict:
        """
        计算粒子在当前时刻的动态状态（双曲坐标、当前速度、当前温度）
        
        粒子距离原点随时间增加：距离 = 初始距离 + 速度 × 时间
        
        性能优化：直接接收原始数值，避免构建 Point 对象的开销。
        
        Args:
            vec: 情感向量（归一化后的方向向量），支持 numpy.ndarray 或 torch.Tensor
            v: 初始速度/强度
            T: 初始温度
            born: 生成时间戳
            t_now: 当前时间戳
            weight: 粒子质量（初始情绪向量的模长），默认 1.0
            
        Returns:
            包含以下键的字典:
            - current_vector: 庞加莱球坐标 (torch.Tensor)
            - current_v: 当前速度/强度 (float)
            - current_T: 当前温度 (float)
            - distance_from_origin: 距离原点的距离（双曲距离，float）
            - is_expired: 是否已消失（超过最大半径，bool）
        """
        # 类型转换：numpy.ndarray -> torch.Tensor
        if isinstance(vec, np.ndarray):
            vec_tensor = torch.from_numpy(vec).float()
        elif isinstance(vec, torch.Tensor):
            vec_tensor = vec.float()
        else:
            raise TypeError(f"Unsupported vector type: {type(vec)}, expected numpy.ndarray or torch.Tensor")
        
        # 1. 物理演化：计算当前时刻的 v 和 T
        v_current = TimePhysics.f(v, born, t_now)
        T_current = TimePhysics.g(T, born, t_now)
        
        # 2. 计算时间差（秒）
        dt = max(0.0, t_now - born)
        
        # 3. 计算当前距离：初始距离 + 速度 × 时间
        # 初始距离由 weight 和 scaling_factor 决定
        initial_distance = weight * self.scaling_factor
        # 距离随时间增加：速度 × 时间（速度是单位时间的距离增量）
        current_distance = initial_distance + v_current * dt
        
        # 4. 检查是否超过最大半径
        is_expired = current_distance >= self.max_radius
        if is_expired:
            # 如果超过最大半径，返回边界上的点
            current_distance = self.max_radius
        
        # 5. 空间投影：欧式 -> 双曲
        # 零向量保护：避免归一化零向量导致的 NaN
        vec_norm = torch.norm(vec_tensor)
        if vec_norm < 1e-9:
            direction = torch.zeros_like(vec_tensor)
        else:
            # vec 已经是归一化的方向向量
            direction = F.normalize(vec_tensor, p=2, dim=-1)
        
        # 广义双曲投影公式
        # r = tanh(sqrt(c) * dist / 2) / sqrt(c)
        # 这里 dist = current_distance（随时间增加）
        arg = self.sqrt_c * current_distance / 2.0
        r = torch.tanh(torch.tensor(arg, dtype=vec_tensor.dtype)) / self.sqrt_c
        
        poincare_coord = r * direction
        
        return {
            "current_vector": poincare_coord,
            "current_v": v_current,
            "current_T": T_current,
            "distance_from_origin": float(current_distance),
            "is_expired": is_expired
        }

