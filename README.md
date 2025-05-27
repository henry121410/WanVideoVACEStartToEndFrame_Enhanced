# ComfyUI-MyWanVideoVACE

这是一个增强版的WanVideo VACE节点包，提供了对原始`WanVideoVACEStartToEndFrame`节点的扩展功能。

## 功能特性

### WanVideo VACE Start To End Frame (Enhanced)

这个增强版节点在原有功能基础上添加了以下特性：

1. **多图像插入支持**：支持同时插入最多6个图像（原始insert_image + 5个编号图像）
2. **独立的帧索引**：每个插入图像都有自己的帧索引设置
3. **独立的遮罩支持**：每个插入图像都可以有自己的遮罩
4. **叠加处理**：支持start_image、end_image、control_images和多个insert_images的叠加处理
5. **正确的混合逻辑**：白色遮罩区域显示插入图像，黑色区域显示原有内容

## 输入参数

### 必需参数
- `num_frames`: 要生成的帧数
- `empty_frame_level`: 空帧的级别（0.0 = 黑色，1.0 = 白色）

### 可选参数
- `start_image`: 序列的起始图像
- `end_image`: 序列的结束图像
- `control_images`: 控制图像序列
- `inpaint_mask`: 修复遮罩

### 插入图像参数
- `insert_image`: 主要插入图像
- `insert_frame_index`: 主要插入图像的帧索引
- `insert_mask_to_insert_image`: 主要插入图像的遮罩

- `insert_image_num1` 到 `insert_image_num5`: 额外的插入图像
- `insert_frame_num1_index` 到 `insert_frame_num5_index`: 对应的帧索引
- `insert_mask_to_insert_image_num1` 到 `insert_mask_to_insert_image_num5`: 对应的遮罩

## 使用方法

1. 将此文件夹放置在ComfyUI的`custom_nodes`目录下
2. 重启ComfyUI
3. 在节点菜单中找到"WanVideoWrapper/Enhanced"分类
4. 使用"WanVideo VACE Start To End Frame (Enhanced)"节点

## 混合逻辑

- **白色遮罩区域**：显示插入的图像
- **黑色遮罩区域**：显示原有的帧内容
- **灰色遮罩区域**：按比例混合两者

## 注意事项

1. 这是一个独立的节点包，不会被原作者的更新覆盖
2. 所有图像会自动调整大小以匹配目标尺寸
3. 帧索引会自动限制在有效范围内
4. 支持alpha通道作为遮罩，也可以使用独立的遮罩输入

## 兼容性

- 兼容ComfyUI和WanVideoWrapper
- 需要PyTorch和相关依赖
- 输出格式与原始节点兼容

## 更新日志

### v1.0
- 初始版本
- 支持多图像插入
- 修复了原始节点的混合逻辑问题
- 添加了叠加处理功能 