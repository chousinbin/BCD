## 项目难点

- 电池图片容易搜集，但是互联网上现存的数据集大都是垃圾分类数据集，没有对电池的类别进行细分，需要团队手动标注；
- 在对重叠堆放的电池进行目标检测时，被压住的电池可能无法被识别；
- 在标注时分类标准不明确，同种锂电池有单个柱状的还有多个柱状锂电池包还有片状的锂电池，标注标准可能影响训练结果；

## 数据标注

### 分类依据

| 标签名 |        解释        | 编号 |
| :----: | :----------------: | :--: |
|  bike  | 电动自行车手提电池 |  0   |
| button |      纽扣电池      |  1   |
|  car   | 汽车、摩托车蓄电池 |  2   |
|  dry   |       干电池       |  3   |
| laptop |     笔记本电池     |  4   |
|   li   |     柱形锂电池     |  5   |
| phone  |      手机电池      |  6   |
|  toy   |    9V 玩具电池     |  7   |

### 标注技巧

#### 查看文件夹下 json 文件的个数

```shell
Get-ChildItem -Path "D:\YOLO\unlabeled" -Filter *.json -Recurse | Measure-Object | Select-Object -ExpandProperty Count
```

## 数据整理

### 整理思路

1. 检查文件夹，把未被标注的图片删除；
2. 把 json 文件转换为 txt 文件；
3. 按照 YOLO 格式分类存放。

### 删除未标注的图片

```py
import os

def remove_images_without_json(image_folder):
    # 获取文件夹中所有文件的列表
    files = os.listdir(image_folder)
    
    # 分离出图片文件和JSON文件
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    json_files = [f for f in files if f.endswith('.json')]
    
    # 遍历每个图片文件，检查是否有对应的JSON文件
    for image_file in image_files:
        image_name, _ = os.path.splitext(image_file)  # 去掉扩展名
        corresponding_json = image_name + '.json'
        
        # 如果没有找到对应的json文件，删除该图片
        if corresponding_json not in json_files:
            image_path = os.path.join(image_folder, image_file)
            print(f"删除没有对应JSON的图片文件: {image_path}")
            os.remove(image_path)

# 示例用法
image_folder = 'D:\Github\BCD\dataset'  # 替换为你的文件夹路径
remove_images_without_json(image_folder)
```

### json2txt

```py
import os
import json

# 配置文件路径
json_dir = 'D:\Github\BCD\dataset\jsons'  # 替换为你的Labelme JSON文件所在目录路径
output_dir = 'D:\Github\BCD\dataset\labels'  # 替换为输出YOLOv5格式的TXT文件的目录路径

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 定义你的标签到类别ID的映射表
label_to_id_mapping = {
    'bike': 0,
    'button': 1,
    'car': 2,
    'dry': 3,
    'laptop': 4,
    'li': 5,
    'phone': 6,
    'toy': 7,
}


# 定义函数，将多边形转为YOLO格式的边界框
def convert_polygon_to_yolo(size, polygon):
    x_min = min([p[0] for p in polygon])
    x_max = max([p[0] for p in polygon])
    y_min = min([p[1] for p in polygon])
    y_max = max([p[1] for p in polygon])
    
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (x_min + x_max) / 2.0 * dw
    y_center = (y_min + y_max) / 2.0 * dh
    width = (x_max - x_min) * dw
    height = (y_max - y_min) * dh
    
    return (x_center, y_center, width, height)

# 遍历所有的JSON文件
for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        json_path = os.path.join(json_dir, filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        
        output_txt_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
        
        with open(output_txt_path, 'w') as out_file:
            for shape in data['shapes']:
                label = shape['label']
                
                # 获取类ID
                if label in label_to_id_mapping:
                    class_id = label_to_id_mapping[label]
                else:
                    print(f"Warning: label '{label}' not in label_to_id_mapping. Skipping.")
                    continue
                
                points = shape['points']
                bbox = convert_polygon_to_yolo((image_width, image_height), points)
                
                out_file.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

print("所有的JSON文件已经成功转换为YOLOv5的TXT格式！")

```

