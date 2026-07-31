# MaixCAM Vision UART / G-code Output

本文档说明视觉端发给电控端 / 写字机的串口输出格式。

当前代码位置：

- `maixcam/main.py`
- `SERIAL_PORT = "/dev/ttyS4"`
- `SERIAL_BAUDRATE = 115200`
- `SERIAL_OUTPUT_FORMAT = "binary"`

当前默认输出是二进制任务包。G-code 输出仍保留在代码里，设置 `SERIAL_OUTPUT_FORMAT = "gcode"` 后可切换到写字机 G-code。

当前使用 MaixCAM2 的 UART4：

| MaixCAM2 | 功能 | 连接到 STM32 |
| --- | --- | --- |
| `A21` | `UART4_TX` | STM32 RX |
| `A22` | `UART4_RX` | STM32 TX |
| `GND` | GND | STM32 GND |

如果只需要视觉端单向发送，至少连接 `A21 -> STM32 RX` 和 `GND -> STM32 GND`。

发送时机：

- 控制台会周期性打印当前识别结果，便于视觉端调试。
- 串口二进制包不会周期性发送。
- 每次短按 MaixCAM2 按键，视觉端只发送一次当前任务包。
- 长按按键仍用于退出程序。

## 可选 G-code 输出

每块拼图会输出一组 G-code：

```text
G21
G90
; piece A rot=-10.5/shape
G1 Z5 F3000
G0 X12 Y46 F5000
G1 Z0 F3000
G1 A-10.5 F3000
G1 X80 Y90 F3000
G1 Z5 F3000
```

相关参数在 `maixcam/main.py` 顶部：

```python
GCODE_FEEDRATE = 3000
GCODE_TRAVEL_FEEDRATE = 5000
GCODE_PEN_UP_Z = 5.0
GCODE_PEN_DOWN_Z = 0.0
GCODE_ROTATE_AXIS = "A"
```

如果写字机固件没有旋转轴，把 `GCODE_ROTATE_AXIS = ""`，视觉端会保留注释里的角度，但不会发送 `G1 A...`。

## 总体格式

以下是当前默认二进制模式格式，当 `SERIAL_OUTPUT_FORMAT = "binary"` 时使用。

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

当前视觉端已经按 STM32 坐标系交换 X/Y 后再发送：

```text
STM32_X = VisionMech_Y
STM32_Y = VisionMech_X
```

也就是说，电控端按收到的 `PICK_X/PICK_Y/PLACE_X/PLACE_Y` 直接使用即可，不需要再交换。

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
