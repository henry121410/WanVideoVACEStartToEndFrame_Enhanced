# ComfyUI-MyWanVideoVACE

An enhanced version of the WanVideo VACE node package, providing extended functionality to the original `WanVideoVACEStartToEndFrame` node.
This project is an enhanced version of a node from the [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) repository.
这是一个增强版的WanVideo VACE节点包，提供了对原始`WanVideoVACEStartToEndFrame`节点的扩展功能。

## Features
## 功能特性

### WanVideo VACE Start To End Frame (Enhanced)

This enhanced node adds the following features on top of the original functionality:
这个增强版节点在原有功能基础上添加了以下特性：

1. **Multiple Image Insertion Support**: Supports inserting up to 6 images simultaneously (original insert_image + 5 numbered images)
1. **多图像插入支持**：支持同时插入最多6个图像（原始insert_image + 5个编号图像）
2. **Independent Frame Indices**: Each inserted image has its own frame index setting
2. **独立的帧索引**：每个插入图像都有自己的帧索引设置
3. **Independent Mask Support**: Each inserted image can have its own mask
3. **独立的遮罩支持**：每个插入图像都可以有自己的遮罩
4. **Overlay Processing**: Supports overlay processing of start_image, end_image, control_images, and multiple insert_images
4. **叠加处理**：支持start_image、end_image、control_images和多个insert_images的叠加处理
5. **Correct Blending Logic**: White mask areas display the inserted image, while black areas display the original content
5. **正确的混合逻辑**：白色遮罩区域显示插入图像，黑色区域显示原有内容

## Input Parameters
## 输入参数

### Required Parameters
### 必需参数
- `num_frames`: Number of frames to generate
- `num_frames`: 要生成的帧数
- `empty_frame_level`: Level of empty frames (0.0 = black, 1.0 = white)
- `empty_frame_level`: 空帧的级别（0.0 = 黑色，1.0 = 白色）

### Optional Parameters
### 可选参数
- `start_image`: Starting image of the sequence
- `start_image`: 序列的起始图像
- `end_image`: Ending image of the sequence
- `end_image`: 序列的结束图像
- `control_images`: Control image sequence
- `control_images`: 控制图像序列
- `inpaint_mask`: Inpaint mask
- `inpaint_mask`: 修复遮罩

### Insert Image Parameters
### 插入图像参数
- `insert_image`: Primary inserted image
- `insert_image`: 主要插入图像
- `insert_frame_index`: Frame index for the primary inserted image
- `insert_frame_index`: 主要插入图像的帧索引
- `insert_mask_to_insert_image`: Mask for the primary inserted image
- `insert_mask_to_insert_image`: 主要插入图像的遮罩

- `insert_image_num1` to `insert_image_num5`: Additional inserted images
- `insert_image_num1` 到 `insert_image_num5`: 额外的插入图像
- `insert_frame_num1_index` to `insert_frame_num5_index`: Corresponding frame indices
- `insert_frame_num1_index` 到 `insert_frame_num5_index`: 对应的帧索引
- `insert_mask_to_insert_image_num1` to `insert_mask_to_insert_image_num5`: Corresponding masks
- `insert_mask_to_insert_image_num1` 到 `insert_mask_to_insert_image_num5`: 对应的遮罩

## Usage
## 使用方法

1. Place this folder in the `custom_nodes` directory of ComfyUI
1. 将此文件夹放置在ComfyUI的`custom_nodes`目录下
2. Restart ComfyUI
2. 重启ComfyUI
3. Find the "WanVideoWrapper/Enhanced" category in the node menu
3. 在节点菜单中找到"WanVideoWrapper/Enhanced"分类
4. Use the "WanVideo VACE Start To End Frame (Enhanced)" node
4. 使用"WanVideo VACE Start To End Frame (Enhanced)"节点

## Blending Logic
## 混合逻辑

- **White mask areas**: Display the inserted image
- **白色遮罩区域**：显示插入的图像
- **Black mask areas**: Display the original frame content
- **黑色遮罩区域**：显示原有的帧内容
- **Gray mask areas**: Blend both proportionally
- **灰色遮罩区域**：按比例混合两者

## Notes
## 注意事项

1. This is an independent node package and will not be overwritten by updates from the original author
1. 这是一个独立的节点包，不会被原作者的更新覆盖
2. All images will be automatically resized to match the target dimensions
2. 所有图像会自动调整大小以匹配目标尺寸
3. Frame indices will be automatically limited to the valid range
3. 帧索引会自动限制在有效范围内
4. Supports alpha channel as mask, and can also use independent mask input
4. 支持alpha通道作为遮罩，也可以使用独立的遮罩输入

## Compatibility
## 兼容性

- Compatible with ComfyUI and WanVideoWrapper
- 兼容ComfyUI和WanVideoWrapper
- Requires PyTorch and related dependencies
- 需要PyTorch和相关依赖
- Output format is compatible with the original node
- 输出格式与原始节点兼容

## Changelog
## 更新日志

### v1.0
### v1.0
- Initial version
- 初始版本
- Added multi-image insertion support
- 支持多图像插入
- Fixed blending logic issue in the original node
- 修复了原始节点的混合逻辑问题
- Added overlay processing functionality
- 添加了叠加处理功能 