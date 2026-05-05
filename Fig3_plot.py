import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
style = dict(size=20, color='black')
L_orbital = ["0","95","0","95"]
plt.rcParams['figure.dpi'] = 600
plt.rcParams['font.size'] = 16
lab1 = ["(a)", "(d)", "(b)", "(e)"]
lab2 = ["(c)", "(f)"]
def read_fortran_binary_complex(filename):
    """
    读取复杂结构的Fortran二进制文件
    
    文件结构:
    第一行: 一个整数, 一个双精度浮点数, 三个整数(nx, ny, nz)
    第二行: 三个双精度浮点数, 三个一维双精度数组(长度分别为nx, ny, nz)
    第三行: 一个字符串(长度10), 两个逻辑变量
    第四行: 一个四维双精度数组, 形状为(nx, ny, nz, 2)
    """
    
    with open(filename, 'rb') as f:
        # ===== 第一行读取 =====
        # 读取第一条记录的长度标记
        rec1_start = np.fromfile(f, dtype=np.int32, count=1)[0]
        print(f"第一行记录长度: {rec1_start} 字节")
        
        # 读取第一行数据: 1个整数 + 1个双精度 + 3个整数
        iter = np.fromfile(f, dtype=np.int32, count=1)[0]
        time = np.fromfile(f, dtype=np.float64, count=1)[0]
        nx, ny, nz = np.fromfile(f, dtype=np.int32, count=3)
        
        print(f"第一行数据: 整数={iter}, 双精度={time}, nx={nx}, ny={ny}, nz={nz}")
        
        # 读取第一条记录的结束标记
        rec1_end = np.fromfile(f, dtype=np.int32, count=1)[0]
        if rec1_start != rec1_end:
            print(f"警告: 第一行记录标记不匹配! 开始: {rec1_start}, 结束: {rec1_end}")
        
        # ===== 第二行读取 =====
        # 读取第二条记录的长度标记
        rec2_start = np.fromfile(f, dtype=np.int32, count=1)[0]
        print(f"第二行记录长度: {rec2_start} 字节")
        
        # 读取三个双精度浮点数
        dx, dy, dz, wxyz = np.fromfile(f, dtype=np.float64, count=4)
        print(f"第二行前三个双精度数: {dx}, {dy}, {dz}, {wxyz}")
        
        # 读取三个一维数组
        # 第一个数组，长度为nx
        xx = np.fromfile(f, dtype=np.float64, count=nx)
        # 第二个数组，长度为ny
        yy = np.fromfile(f, dtype=np.float64, count=ny)
        # 第三个数组，长度为nz
        zz = np.fromfile(f, dtype=np.float64, count=nz)
        
        print(f"第二行数组形状: array1={xx.shape}, array2={yy.shape}, array3={zz.shape}")
        
        # 读取第二条记录的结束标记
        rec2_end = np.fromfile(f, dtype=np.int32, count=1)[0]
        if rec2_start != rec2_end:
            print(f"警告: 第二行记录标记不匹配! 开始: {rec2_start}, 结束: {rec2_end}")
        
        # ===== 第三行读取 =====
        # 读取第三条记录的长度标记
        rec3_start = np.fromfile(f, dtype=np.int32, count=1)[0]
        print(f"第三行记录长度: {rec3_start} 字节")
        
        # 读取字符串(10个字符)
        # 使用struct读取固定长度的字符串
        string_bytes = f.read(10)
        stored_name1 = string_bytes.decode('ascii').rstrip('\x00')  # 移除填充的空字符
        
        # 读取两个逻辑变量
        # Fortran中逻辑变量通常存储为整数(4字节)或字节(1字节)
        # 这里尝试4字节整数
        logical1 = np.fromfile(f, dtype=np.int32, count=1)[0]
        logical2 = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # 转换为Python布尔值 (Fortran中非零通常表示True)
        vector1 = logical1 != 0
        isospin1 = logical2 != 0
        
        print(f"第三行数据: 字符串='{stored_name1}', 逻辑1={vector1}, 逻辑2={isospin1}")
        
        # 读取第三条记录的结束标记
        rec3_end = np.fromfile(f, dtype=np.int32, count=1)[0]
        if rec3_start != rec3_end:
            print(f"警告: 第三行记录标记不匹配! 开始: {rec3_start}, 结束: {rec3_end}")
        
        # ===== 第四行读取 =====
        # 读取第四条记录的长度标记
        rec4_start = np.fromfile(f, dtype=np.int32, count=1)[0]
        print(f"第四行记录长度: {rec4_start} 字节")
        
        # 计算四维数组的总元素数
        total_elements = nx * ny * nz * 2
        
        # 读取四维数组数据
        rho_flat = np.fromfile(f, dtype=np.float64, count=total_elements)
        
        # 重塑为四维数组 (nx, ny, nz, 2)，使用Fortran列优先顺序
        rho = rho_flat.reshape((nx, ny, nz, 2), order='F')
        
        print(f"第四行四维数组形状: {rho.shape}")
        
        # 读取第四条记录的结束标记
        rec4_end = np.fromfile(f, dtype=np.int32, count=1)[0]
        if rec4_start != rec4_end:
            print(f"警告: 第四行记录标记不匹配! 开始: {rec4_start}, 结束: {rec4_end}")

        # ===== 第五行读取 =====
        # 读取第五条记录的长度标记
        rec5_start = np.fromfile(f, dtype=np.int32, count=1)[0]
        print(f"第五行记录长度: {rec5_start} 字节")
        
        # 读取字符串(10个字符)
        # 使用struct读取固定长度的字符串
        string_bytes = f.read(10)
        stored_name2 = string_bytes.decode('ascii').rstrip('\x00')  # 移除填充的空字符
        
        # 读取两个逻辑变量
        # Fortran中逻辑变量通常存储为整数(4字节)或字节(1字节)
        # 这里尝试4字节整数
        logical1 = np.fromfile(f, dtype=np.int32, count=1)[0]
        logical2 = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # 转换为Python布尔值 (Fortran中非零通常表示True)
        vector2 = logical1 != 0
        isospin2 = logical2 != 0
        
        print(f"第五行数据: 字符串='{stored_name2}', 逻辑1={vector2}, 逻辑2={isospin2}")
        
        # 读取第五条记录的结束标记
        rec5_end = np.fromfile(f, dtype=np.int32, count=1)[0]
        if rec5_start != rec5_end:
            print(f"警告: 第五行记录标记不匹配! 开始: {rec5_start}, 结束: {rec5_end}")

        # ===== 第六行读取 =====
        # 读取第六条记录的长度标记
        rec6_start = np.fromfile(f, dtype=np.int32, count=1)[0]
        print(f"第六行记录长度: {rec6_start} 字节")
        
        # 计算五维数组的总元素数
        total_elements = nx * ny * nz * 3 * 2
        
        # 读取五维数组数据
        current_flat = np.fromfile(f, dtype=np.float64, count=total_elements)
        
        # 重塑为五维数组 (nx, ny, nz, 3, 2)，使用Fortran列优先顺序
        current = current_flat.reshape((nx, ny, nz, 3, 2), order='F')
        
        print(f"第六行五维数组形状: {current.shape}")
        
        # 读取第六条记录的结束标记
        rec6_end = np.fromfile(f, dtype=np.int32, count=1)[0]
        if rec6_start != rec6_end:
            print(f"警告: 第六行记录标记不匹配! 开始: {rec6_start}, 结束: {rec6_end}")

        # 返回所有数据
        return {
            'line1': {
                'integer': iter,
                'double': time,
                'nx': nx,
                'ny': ny,
                'nz': nz
            },
            'line2': {
                'doubles': [dx, dy, dz, wxyz],
                'xx': xx,
                'yy': yy,
                'zz': zz
            },
            'line3': {
                'string': stored_name1,
                'logical1': vector1,
                'logical2': isospin1
            },
            'line4': {
                'rho': rho
            },
            'line5': {
                'string': stored_name2,
                'logical1': vector2,
                'logical2': isospin2
            }, 
            'line6': {
                'current': current
            }                 
        }
def compute_center_of_mass(rho, x, y, z,n):
    rho_total = rho[:, :, :, n]# + rho[:, :, :, 1]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    total_mass = np.sum(rho_total)
    if total_mass == 0:
        return (0.0, 0.0, 0.0)
    cx = np.sum(X * rho_total) / total_mass
    cy = np.sum(Y * rho_total) / total_mass
    cz = np.sum(Z * rho_total) / total_mass
    return (cx, cy, cz)

def rotate_field_scipy(rho, x, z, angle_deg, center):
    nx, ny, nz = rho.shape
    cx, cy, cz = center
    ix = np.arange(nx)
    iz = np.arange(nz)
    IX, IZ = np.meshgrid(ix, iz, indexing='ij')
    X_phys = x[IX] - cx
    Z_phys = z[IZ] - cz
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    X_rot = cos_t * X_phys - sin_t * Z_phys
    Z_rot = sin_t * X_phys + cos_t * Z_phys
    X_new = X_rot + cx
    Z_new = Z_rot + cz
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    ix_new = (X_new - x[0]) / dx
    iz_new = (Z_new - z[0]) / dz
    rho_rotated = np.empty_like(rho)
    for iy in range(ny):
        slice_2d = rho[:, iy, :]
        rho_rotated[:, iy, :] = map_coordinates(
            slice_2d, [ix_new, iz_new], order=3, mode='nearest'
        )
    rho[:] = rho_rotated
# 定义要处理的文件列表
files = [
    '003850(T100).tdd', 
    '002400(T100).tdd',
    '003500(T000).tdd', 
    '002650(T000).tdd',
    '002850(175).tdd',
    '003710.tdd'
]
slope = [0.359, 0.5979, 0.7801, 0.9127, 0.9949, 1.046]
# 创建2×3的子图布局
fig, axes = plt.subplots(3, 2, figsize=(12, 18))
axes = axes.flatten()  # 将2D数组展平为1D以便于迭代

# 首先收集所有子图的当前幅度数据，用于确定统一的颜色范围
all_current_magnitudes = []
# 存储所有数据用于计算全局范围
all_data = []
# 第一步：收集所有数据并计算统一的颜色范围
#for i, filename in enumerate(files):
for i in range(4):
    try:
        filename=files[i]
        print(f"正在收集数据以确定颜色范围: {filename}")
        data = read_fortran_binary_complex(filename)
        
        # 提取数据
        nx, ny, nz = data['line1']['nx'], data['line1']['ny'], data['line1']['nz']
        x1, y1, z1 = data['line2']['xx'], data['line2']['yy'], data['line2']['zz']
        Z, X = np.meshgrid(x1, z1)
        rho = data['line4']['rho']
        current = data['line6']['current']
        
        # 计算当前分量和密度
        rhon1 = (rho[:, int(ny/2)+1, :, 0] + rho[:, int(ny/2), :, 0])*0.5
        rhop1 = (rho[:, int(ny/2)+1, :, 1] + rho[:, int(ny/2), :, 1])*0.5  
        rho1 = rhon1 + rhop1

        currentxn = (current[:, int(ny/2)+1, :, 0, 0] + current[:, int(ny/2), :, 0, 0])*0.5 * 2000
        currentxp = (current[:, int(ny/2)+1, :, 0, 1] + current[:, int(ny/2), :, 0, 1])*0.5 * 2000
        currentx = currentxn + currentxp

        currentzn = (current[:, int(ny/2)+1, :, 2, 0] + current[:, int(ny/2), :, 2, 0])*0.5 * 2000
        currentzp = (current[:, int(ny/2)+1, :, 2, 1] + current[:, int(ny/2), :, 2, 1])*0.5 * 2000
        currentz = currentzn + currentzp
        
        # 计算当前幅度
        current_magnitude1 = np.sqrt(currentx**2 + currentz**2)
        current_magnitude = np.log(np.sqrt(currentx**2 + currentz**2))
        all_current_magnitudes.append(current_magnitude)
            # 存储数据以便后续处理
        currentx = currentx/current_magnitude1*current_magnitude
        currentz = currentz/current_magnitude1*current_magnitude

        all_data.append({
            'filename': filename,
            'X': X,
            'Z': Z,
            'rho1': rho1,
            'currentx': currentx,
            'currentz': currentz,
            'current_magnitude': current_magnitude,
            'nx': nx,
            'ny': ny,
            'nz': nz
        })    
        print(f"完成收集: {filename}")
        
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}")        
        all_data.append({
            'filename': filename,
            'X': None,
            'Z': None,
            'rho1': None,
            'currentx': None,
            'currentz': None,
            'current_magnitude': None,
            'nx': 0,
            'ny': 0,
            'nz': 0
        })
        # 如果文件无法读取，添加空数组
        all_current_magnitudes.append(np.array([]))
# 计算全局范围
if all_data:
    # 计算密度的全局范围
    rho_min_values = []
    rho_max_values = []
    
    # 计算电流幅度的全局范围
    current_min_values = []
    current_max_values = []
    
    # 计算箭头的最大长度（用于统一scale参数）
    arrow_max_lengths = []
    
    for data in all_data:
        if data['rho1'] is not None:
            rho_min_values.append(np.nanmin(data['rho1']))
            rho_max_values.append(np.nanmax(data['rho1']))
        
        if data['current_magnitude'] is not None:
            current_min_values.append(np.nanmin(data['current_magnitude']))
            current_max_values.append(np.nanmax(data['current_magnitude']))
            
            # 计算箭头的最大长度（用于确定统一的scale）
            if data['currentx'] is not None and data['currentz'] is not None:
                # 计算所有箭头的长度
                arrow_lengths = np.sqrt(data['currentx']**2 + data['currentz']**2)
                arrow_max_lengths.append(np.nanmax(arrow_lengths))
    
    # 获取全局范围
    if rho_min_values:
        global_rho_min = np.min(rho_min_values)
        global_rho_max = np.max(rho_max_values)
        print(f"密度全局范围: {global_rho_min:.4f} 到 {global_rho_max:.4f}")
    else:
        global_rho_min = 0.0
        global_rho_max = 1.0
    
    if current_min_values:
        global_current_min = np.min(current_min_values)
        global_current_max = np.max(current_max_values)
        print(f"电流幅度全局范围: {global_current_min:.4f} 到 {global_current_max:.4f}")
    else:
        global_current_min = 0.0
        global_current_max = 1.0
    
    # 计算统一的箭头scale参数
    if arrow_max_lengths:
        global_arrow_max = np.max(arrow_max_lengths)
        # 根据最大箭头长度调整scale值，使得箭头在子图中显示合适
        # scale值越大，箭头越短；scale值越小，箭头越长
        # 这里我们基于最大箭头长度和图形尺寸来计算一个合适的scale
        # 经验公式：scale ≈ 图形尺寸 / (最大箭头长度 * 某个因子)
        # 我们可以先计算一个基准值，然后根据需要进行调整
        base_scale = 200 / global_arrow_max if global_arrow_max > 0 else 25
        # 限制scale的范围，避免太大或太小
        unified_scale = max(min(base_scale, 50), 0.1)
        print(f"全局最大箭头长度: {global_arrow_max:.4f}")
        print(f"统一箭头scale参数: {unified_scale:.2f}")
    else:
        unified_scale = 25
    
    # 设置统一的密度等高线级别
    # 使用全局范围创建等间距的等高线级别
    num_contour_levels = 6  # 包括最小值和最大值，所以实际级别数为num_contour_levels-1
    global_rho_levels = np.linspace(global_rho_min, global_rho_max, num_contour_levels)
    print(f"统一密度等高线级别: {global_rho_levels}")
    
else:
    global_rho_min = 0.0
    global_rho_max = 1.0
    global_current_min = 0.0
    global_current_max = 1.0
    unified_scale = 25
    global_rho_levels = np.linspace(0, 1, 6)
# 计算统一的颜色范围（所有子图的最大值）
if all_current_magnitudes:
    # 计算所有非空数组的最大值
    valid_magnitudes = [arr for arr in all_current_magnitudes if arr.size > 0]
    if valid_magnitudes:
        # 获取所有有效数组的最大值
        max_values = [np.nanmax(arr) for arr in valid_magnitudes]
        global_max = np.max(max_values)
        print(f"统一的颜色范围最大值: {global_max}")
    else:
        global_max = 1.0
else:
    global_max = 1.0

# 第二步：绘制所有子图
for ii, data_dict in enumerate(all_data):
    try:
        filename = data_dict['filename']
        print(f"正在绘制文件: {filename}")
        if data_dict['X'] is None or data_dict['rho1'] is None:
            raise ValueError(f"数据为空，无法绘制: {filename}")        
        
        #data = read_fortran_binary_complex(filename)
        
        # 提取数据
        X = data_dict['X']
        Z = data_dict['Z']
        rho1 = data_dict['rho1']
        currentx = data_dict['currentx']
        currentz = data_dict['currentz']
        current_magnitude = data_dict['current_magnitude']   
        
        # 创建掩码：只显示大于阈值1%的箭头
        threshold = current_magnitude.max() * 0.00005 if current_magnitude.max() > 0 else 0
        masked_currentx = np.where(current_magnitude > threshold, currentx, np.nan)
        masked_currentz = np.where(current_magnitude > threshold, currentz, np.nan)
        masked_current_magnitude = np.where(current_magnitude > threshold, current_magnitude, np.nan)
        
        
        i=ii       
        ax = axes[i]
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['right'].set_linewidth(2.0)
        # 1. 首先绘制密度等高线图（放在底层）
        contour_levels = 4
        '''
        # 使用contourf填充颜色作为背景
        contour_fill = ax.contourf(X, Z, rho1, 
                                   levels=contour_levels, 
                                   cmap='Greys',  # 使用灰色系，这样箭头颜色会更明显
                                   alpha=0.5,     # 设置透明度，避免遮盖箭头
                                   vmin=global_rho_min,  # 统一的最小值
                                   vmax=global_rho_max,  # 统一的最大值
                                   zorder=1)      # 设置zorder为1，确保在底层                           
        '''
        # 确定当前子图的rho1的最小最大值
        rho_min_local = rho1.min()+rho1.max()/3.1
        rho_max_local = rho1.max()
        # 生成级别
        levels = np.linspace(rho_min_local, rho_max_local, contour_levels)
        # 生成颜色：从深到浅，使用灰度颜色映射，数值从0.2到0.8（避免纯黑和纯白）
        colors = plt.cm.Greys(np.linspace(0.8, 0.2, contour_levels))
        # 生成线宽：从粗到细，例如2.0到0.5
        linewidths = np.linspace(2.0, 1.2, contour_levels)

        # 绘制等高线
        contour_lines = ax.contour(X, Z, rho1, levels=levels, colors=colors, linewidths=linewidths, alpha=0.8, zorder=2)  
        '''      
        # 添加等高线线条
        contour_lines = ax.contour(X, Z, rho1, 
                                   levels=contour_levels, 
                                   colors='dimgray',
                                   linewidths=2,
                                   alpha=0.8,
                                   zorder=2)      # 线条在填充之上
        '''
        # 2. 然后绘制箭头图（叠加在等高线图上，使用统一的颜色范围）
        quiver = ax.quiver(X, Z, 
                          masked_currentx,  # X分量
                          masked_currentz,  # Z分量
                          masked_current_magnitude,  # 使用掩码后的幅度
                          cmap='coolwarm', 
                          scale=25,#unified_scale,#25, 
                          width=0.004,
                          alpha=0.8,
                          zorder=3,         # 设置zorder为3，确保在最上层
                          clim=(0, global_max))  # 设置统一的颜色范围
        

        if ii == 1:   
            arrow = FancyArrowPatch(
                (3, -14.5),
                (13, -8.5),
                arrowstyle='->',
                mutation_scale=20,
                color='dimgrey',
                linewidth=3,
                connectionstyle="arc3,rad=0.3"  # 控制弯曲程度
            )
            ax.add_patch(arrow)    
            arrow2 = FancyArrowPatch(
                (-5, 18),
                (-13, 10),
                arrowstyle='->',
                mutation_scale=20,
                color='dimgrey',
                linewidth=3,
                connectionstyle="arc3,rad=0.4"  # 控制弯曲程度
            )
            ax.add_patch(arrow2)
            ax.arrow(2.0, 3.2, 2.2, 1.7, head_width=0.5, head_length=0.8, linewidth=2.0, fc='grey', ec='grey')
        elif ii == 3:   
            arrow = FancyArrowPatch(
                (3, -14.0),
                (14, -9.0),
                arrowstyle='->',
                mutation_scale=20,
                color='dimgrey',
                linewidth=3,
                connectionstyle="arc3,rad=0.4"  # 控制弯曲程度
            )
            ax.add_patch(arrow)    
            arrow2 = FancyArrowPatch(
                (-5, 17.0),
                (-14, 10),
                arrowstyle='->',
                mutation_scale=20,
                color='dimgrey',
                linewidth=3,
                connectionstyle="arc3,rad=0.45"  # 控制弯曲程度
            )
            ax.add_patch(arrow2)                                      
            ax.arrow(2.0, 3.2, 2.2, 1.7, head_width=0.5, head_length=0.8, linewidth=2.0, fc='grey', ec='grey')
        # 设置子图标题和标签
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)       
        if i==0 or i==2:   
           ax.set_ylabel('Z (fm)', fontsize=18)
        ax.grid(True, alpha=0.3)
        
        # 添加文本标签
        if i <= 1:
            ax.text(12.0, 16.0, s='T=1.0', ha='center', **style)
        else:
            ax.text(12.0, 16.0, s='T=0.0', ha='center', **style)  
        ax.text(0.05,0.9,lab1[i], transform=axes[i].transAxes, **style)     
        ax.text(-11.0, -18.0, s=r'$L_{CN}=$'+L_orbital[i], ha='center', **style)
        
        print(f"完成绘制文件: {filename}")
        ax.tick_params(axis='y', pad=8)  # pad参数控制距离，默认值约为4
        ax.tick_params(axis='x', pad=6)  # pad参数控制距离，默认值约为4
        ax.tick_params(which='major', length=7,width=2)   # 主刻度长度
        ax.tick_params(which='minor', length=4,width=2)   # 次刻度长度，比主刻度短
        ax.tick_params(axis='both',  # 应用于x轴和y轴
               which='both',   # 同时应用于主刻度和次刻度
               direction='in')       # 刻度宽度            
    except Exception as e:
        print(f"处理文件 {filename} 时出错: {e}")
        # 如果文件无法读取，显示空白图
        axes[i].set_title(f'{filename} - Error')
        axes[i].text(0.5, 0.5, 'Error loading file', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[i].transAxes)
rho_log_list = [] 
above_data = []          # (X_sub, Z_sub, rho_sub, x_sub, z_sub)
for ii in range(1):
    i=ii+4
    data = read_fortran_binary_complex(files[i])
    nx, ny, nz = data['line1']['nx'], data['line1']['ny'], data['line1']['nz']
    x_arr = data['line2']['xx']
    y_arr = data['line2']['yy']
    z_arr = data['line2']['zz']
    rho = data['line4']['rho']

    # 旋转密度场
    centern = compute_center_of_mass(rho, x_arr, y_arr, z_arr,0)
    centerp = compute_center_of_mass(rho, x_arr, y_arr, z_arr,1)
    angle = math.atan(slope[2]) / (2*math.pi) * 360
    rotate_field_scipy(rho[:,:,:,0], x_arr, z_arr, angle, centern)
    rotate_field_scipy(rho[:,:,:,1], x_arr, z_arr, angle, centerp)

    # 中平面密度（y方向中间两层平均）
    mid = ny // 2
    rho1n = (rho[:, mid, :, 0] + rho[:, mid+1, :, 0]) 
    rho1p = (rho[:, mid, :, 1] + rho[:, mid+1, :, 1]) 

    # 提取子区域：X 索引 16~75（共60点），Z 全部
    x_sub = x_arr[16:86]   # 60个点
    z_sub = z_arr[:]       # 100个点
    rhon_sub = rho1n[16:86, :]
    rhop_sub = rho1p[16:86, :]
    X_sub, Z_sub = np.meshgrid(x_sub, z_sub, indexing='ij')

    # 对数密度（用于显示和计算全局范围）
#    rho_log = np.log10(rho_sub + 1e-10)
#    rho_log_list.append(rho_log)
    above_data.append((X_sub, Z_sub, rhon_sub, x_sub, z_sub))
    above_data.append((X_sub, Z_sub, rhop_sub, x_sub, z_sub))
# 计算前6个密度图的统一对数密度范围
#global_rho_min = min(arr.min() for arr in rho_log_list)
#global_rho_max = max(arr.max() for arr in rho_log_list)
#print(f"统一对数密度范围: {global_rho_min:.4f} ~ {global_rho_max:.4f}")

# ---------- 第二步：处理下面6个文件，计算绝对差值的对数 ----------
log_absdiff_list = []   # 存储对数绝对差值
for j in range(1):
    data = read_fortran_binary_complex(files[5+j])
    nx, ny, nz = data['line1']['nx'], data['line1']['ny'], data['line1']['nz']
    x_arr = data['line2']['xx']
    y_arr = data['line2']['yy']
    z_arr = data['line2']['zz']
    rho = data['line4']['rho']
#    center = compute_center_of_mass(rho, x_arr, y_arr, z_arr)
#    angle = math.atan(slope[j]) / (2*math.pi) * 360
#    rotate_field_scipy(rho, x_arr, z_arr, angle, center)
    mid = ny // 2
    rho1n_below = (rho[:, mid, :, 0] + rho[:, mid+1, :, 0]) 
    rho1p_below = (rho[:, mid, :, 1] + rho[:, mid+1, :, 1])

    # 上面子区域的网格（X 60点，Z 100点）
    X_sub, Z_sub, rhon_above_sub, x_sub, z_sub = above_data[0]
    X_sub, Z_sub, rhop_above_sub, x_sub, z_sub = above_data[1]
    # 构造下面文件的插值函数
    interp_below = RegularGridInterpolator((x_arr, z_arr), rho1n_below,
                                           bounds_error=False, fill_value=0)
    points = np.array([X_sub.ravel(), Z_sub.ravel()]).T
    rho_below_interp = interp_below(points).reshape(X_sub.shape)
    # 差值，取绝对值，再取对数
    #diff = -rho_below_interp + rho_above_sub
    diffn = -rho1n_below+rhon_above_sub
    diffp = -rho1p_below+rhop_above_sub
    ##diff = np.where(diff < 0, 0, diff)
    #abs_diff = np.abs(diff)
    #log_abs_diff = diff#np.log10(diff + 1e-10)
    log_absdiff_list.append(diffn)
    log_absdiff_list.append(diffp)

# 计算后6个对数绝对差值的统一范围
global_logabs_min = min(diffn.min(), diffp.min())
global_logabs_max = max(diffn.max(), diffp.max())
print(f"对数绝对差值统一范围: {global_logabs_min:.4f} ~ {global_logabs_max:.4f}")
for ii in range(2):
    ax = axes[ii+4]
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.text(0.05,0.9,lab2[ii], transform=axes[ii+4].transAxes, **style) 
    j = ii
    X_sub, Z_sub, rho_above_sub, x_sub, z_sub = above_data[j]
    log_abs_diff = log_absdiff_list[j]
    im = ax.pcolormesh(X_sub, Z_sub, log_abs_diff,
                    cmap='seismic', shading='gouraud',
                    vmin=-0.03, vmax=0.03,
                    alpha=0.9, zorder=1)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-18, 22)
    ax.text(12.0, 18.0, s='T=1.0', ha='center', **style)
    if ii==0:
       ax.text(-12.0, -15.0, s='Neutron', ha='center', **style) 
    else:
       ax.text(-12.0, -15.0, s='Proton', ha='center', **style)     
#    ax.text(x_sub.min()+5, z_sub.min()+5, deformations[j],
#            ha='center', fontsize=20, color='black')
    ax.set_xlabel('X (fm)', fontsize=18)
    if ii==0: 
       ax.set_ylabel('Z (fm)', fontsize=18)
    ax.grid(True, alpha=0.3)   
    if ii == 1:   # 最后一个子图添加 colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.05)
        #cbar.set_label('Δρ', fontsize=12)       
# 调整子图间距
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整rect参数为标题留出空间

# 添加总标题
#fig.suptitle('Current vectors with density contours', fontsize=16, y=0.98)

# 添加统一的颜色条（放在图的右侧）
# 创建一个虚拟的ScalarMappable对象用于颜色条
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# 创建统一的颜色条
norm = Normalize(vmin=0, vmax=global_max)
sm = ScalarMappable(norm=norm, cmap='coolwarm')
sm.set_array([])

# 在图的右侧添加颜色条
#cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
#cbar = fig.colorbar(sm, cax=cbar_ax)
#cbar.set_label('Current Magnitude (Unified Scale)', fontsize=12)

plt.savefig("./2dcurrent_25 scalev7.pdf", dpi=600, bbox_inches='tight')
# 显示图形
plt.show()