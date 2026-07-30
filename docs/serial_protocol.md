# MaixCAM Vision UART Protocol

本文档说明视觉端发给电控端的串口数据包格式。

当前代码位置：

- `maixcam/main.py`
- `SERIAL_BAUDRATE = 115200`
- `SERIAL_BINARY_PACKET = True`

## 总体格式

视觉端周期性发送一个二进制数据包，表示当前识别到的全部拼图移动任务。

```text
AA 55 LEN PAYLOAD CHECKSUM
```

字段说明：

| 字段 | 字节数 | 类型 | 说明 |
| --- | ---: | --- | --- |
| `0xAA` | 1 | uint8 | 帧头 1 |
| `0x55` | 1 | uint8 | 帧头 2 |
| `LEN` | 1 | uint8 | `PAYLOAD` 字节数 |
| `PAYLOAD` | `LEN` | bytes | 拼图任务数据 |
| `CHECKSUM` | 1 | uint8 | 校验和 |

校验和计算：

```c
CHECKSUM = (LEN + PAYLOAD所有字节累加和) & 0xFF;
```

注意：校验不包含帧头 `0xAA 0x55`，只包含 `LEN` 和 `PAYLOAD`。

## PAYLOAD 格式

```text
COUNT RECORD0 RECORD1 ... RECORDn
```

字段说明：

| 字段 | 字节数 | 类型 | 说明 |
| --- | ---: | --- | --- |
| `COUNT` | 1 | uint8 | 当前拼图数量 |
| `RECORD` | 11 | bytes | 单块拼图任务 |

长度关系：

```text
LEN = 1 + COUNT * 11
```

示例：

| 拼图数量 | LEN |
| ---: | ---: |
| 1 | 12 |
| 2 | 23 |
| 3 | 34 |
| 4 | 45 |

## 单块拼图 RECORD

每块拼图占 11 字节，全部为小端序。

```text
ID PICK_X PICK_Y PLACE_X PLACE_Y ROTATE_X10
```

| 字段 | 字节数 | 类型 | 单位 | 说明 |
| --- | ---: | --- | --- | --- |
| `ID` | 1 | uint8 | - | 拼图编号 |
| `PICK_X` | 2 | int16 little-endian | 机械坐标单位 | 当前抓取点 X |
| `PICK_Y` | 2 | int16 little-endian | 机械坐标单位 | 当前抓取点 Y |
| `PLACE_X` | 2 | int16 little-endian | 机械坐标单位 | 目标放置点 X |
| `PLACE_Y` | 2 | int16 little-endian | 机械坐标单位 | 目标放置点 Y |
| `ROTATE_X10` | 2 | int16 little-endian | 0.1 degree | 旋转角度乘以 10 |

`ROTATE_X10` 示例：

| 实际角度 | 发送值 |
| ---: | ---: |
| `12.5°` | `125` |
| `-30.0°` | `-300` |
| `0.0°` | `0` |

当前第一题编号：

| 拼图 | ID |
| --- | ---: |
| A | 0 |
| B | 1 |
| C | 2 |
| D | 3 |
| 未知 | 9 |

第二题如果拼图数量或类型不同，电控端应以 `COUNT` 为准循环解析。

## C 结构体参考

```c
#include <stdint.h>

#pragma pack(push, 1)
typedef struct {
    uint8_t id;
    int16_t pick_x;
    int16_t pick_y;
    int16_t place_x;
    int16_t place_y;
    int16_t rotate_x10;
} VisionMoveRecord;
#pragma pack(pop)
```

解析时不要直接假设 MCU 字节序一定匹配；如果不放心，可以手动按小端合成 `int16_t`。

## 接收状态机建议

1. 等待 `0xAA`
2. 等待 `0x55`
3. 读取 `LEN`
4. 读取 `LEN` 字节 `PAYLOAD`
5. 读取 `CHECKSUM`
6. 校验：

```c
uint8_t sum = len;
for (int i = 0; i < len; i++) {
    sum += payload[i];
}
if (sum != checksum) {
    // 丢弃本帧
}
```

7. 检查长度：

```c
uint8_t count = payload[0];
if (len != 1 + count * 11) {
    // 丢弃本帧
}
```

8. 按 `COUNT` 循环解析每个 `RECORD`。

## 示例

假设有 1 块拼图：

```text
ID = 0
PICK_X = 12
PICK_Y = 46
PLACE_X = 80
PLACE_Y = 90
ROTATE_X10 = -105
```

PAYLOAD：

```text
01 00 0C 00 2E 00 50 00 5A 00 97 FF
```

完整帧：

```text
AA 55 0C 01 00 0C 00 2E 00 50 00 5A 00 97 FF 87
```

其中：

```text
LEN = 0C
CHECKSUM = 87
```

## 控制台调试输出

视觉端控制台会打印人可读的任务摘要：

```text
MOVES 1 | A pick=(12,46) place=(80,90) rot=-10.5
```

注意：控制台调试输出不等于串口内容。串口真实发送的仍然是上文定义的二进制 bytes。
