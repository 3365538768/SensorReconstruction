
// 定义 MUX1 (行) 的控制引脚
const int MUX1_S0 = 33;
const int MUX1_S1 = 25;
const int MUX1_S2 = 26;
const int MUX1_S3 = 27;

// 定义 MUX2 (列) 的控制引脚
const int MUX2_S0 = 21;
const int MUX2_S1 = 22;
const int MUX2_S2 = 23;
const int MUX2_S3 = 14;

// 定义 ADC 输入引脚
const int ADC_PIN = 36; // 请根据您的ESP32型号确认正确的ADC引脚，通常是GPIO36

// ADC的分辨率 (ESP32默认是12位，即0-4095)
const float ADC_MAX_VALUE = 4095.0;

void setup() {
  Serial.begin(115200); // 设置波特率，Python脚本也要匹配
  while (!Serial);

  Serial.println("ESP32 sending ADC data for heatmap...");

  // 设置 MUX 控制引脚为输出模式
  pinMode(MUX1_S0, OUTPUT);
  pinMode(MUX1_S1, OUTPUT);
  pinMode(MUX1_S2, OUTPUT);
  pinMode(MUX1_S3, OUTPUT);

  pinMode(MUX2_S0, OUTPUT);
  pinMode(MUX2_S1, OUTPUT);
  pinMode(MUX2_S2, OUTPUT);
  pinMode(MUX2_S3, OUTPUT);

  // 配置ADC
  analogReadResolution(12);
  analogSetPinAttenuation(ADC_PIN, ADC_11db); // 0-3.6V 范围
}

void loop() {
  Serial.println("START_FRAME"); // 标记数据帧的开始

  for (int row = 0; row < 10; row++) {
    for (int col = 0; col < 10; col++) {
      int adcValue = readADCValue(row, col); // 读取指定通路ADC值

      Serial.print(adcValue);
      if (col < 9) {
        Serial.print(","); // 同一行内的值用逗号分隔
      }
    }
    Serial.println(); // 每一行数据结束后换行
  }
  Serial.println("END_FRAME"); // 标记数据帧的结束

  delay(200); // 每200毫秒发送一帧数据，可以调整此值控制刷新率
}

/**
 * @brief 设置Mux的通道
 *
 * @param s0 Mux的S0引脚
 * @param s1 Mux的S1引脚
 * @param s2 Mux的S2引脚
 * @param s3 Mux的S3引脚
 * @param channel 要选择的通道 (0-15)
 */
void setMuxChannel(int s0, int s1, int s2, int s3, int channel) {
  digitalWrite(s0, (channel & 0x01));
  digitalWrite(s1, (channel & 0x02) >> 1);
  digitalWrite(s2, (channel & 0x04) >> 2);
  digitalWrite(s3, (channel & 0x08) >> 3);
}

/**
 * @brief 读取指定行和列的ADC原始值
 *
 * @param row 阵列的行索引 (0-9)
 * @param col 阵列的列索引 (0-9)
 * @return int ADC原始值 (0-4095)
 */
int readADCValue(int row, int col) {
  setMuxChannel(MUX1_S0, MUX1_S1, MUX1_S2, MUX1_S3, row);
  setMuxChannel(MUX2_S0, MUX2_S1, MUX2_S2, MUX2_S3, col);

  delayMicroseconds(100); // 给予MUX切换时间

  return analogRead(ADC_PIN);
}