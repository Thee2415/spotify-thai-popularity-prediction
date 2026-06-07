# spotify-thai-popularity-prediction
# Thai Music Popularity Prediction Using Machine Learning and Dual-Input RNN
คลังเก็บซอร์สโค้ด (Source Code Repository) นี้เป็นส่วนหนึ่งของโครงงานวิจัยระดับปริญญาตรี ภาควิชาคณิตศาสตร์ คณะวิทยาศาสตร์ มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี (KMUTT) โดยมีวัตถุประสงค์เพื่อพัฒนาแบบจำลองคาดการณ์คะแนนความนิยมและสถานะเพลงฮิตล่วงหน้า 14 วัน (6 ขั้นเวลาของการเก็บข้อมูล) บนแพลตฟอร์ม Spotify

## สถาปัตยกรรมและแบบจำลอง (Proposed Methodology)
การวิจัยนี้ใช้เทคนิคการประมวลผลข้อมูลคุณลักษณะร่วมกันระหว่างข้อมูลเชิงพลวัตตามลำดับเวลา (Temporal Data) และข้อมูลเชิงบริบททั่วไป (Static Context) ผ่านแบบจำลองหลัก 2 กลุ่ม:

1. **Classical Machine Learning:** 
   - Logistic Regression
   - Random Forest
   - Extreme Gradient Boosting (XGBoost)
2. **Deep Learning Topology (Late Fusion):**
   - Dual-Input Long Short-Term Memory (LSTM) Neural Network
   - Dual-Input Gated Recurrent Unit (GRU) Neural Network

## ตัวชี้วัดและประสิทธิภาพ (Key Performance Metrics)
* **การจำแนกประเภทเพลงฮิต (Classification):** แบบจำลอง XGBoost สามารถทำนายสถานะเพลงฮิตได้อย่างแม่นยำสูงถึง **98.08%** และให้ค่าพื้นที่ใต้กราฟ ROC (AUC) อยู่ที่ **0.998**
* **การพยากรณ์คะแนนความนิยม (Regression):** แบบจำลองอิงโครงสร้างต้นไม้ (XGBoost) ทำหน้าที่ได้ดีที่สุดโดยให้ค่า $R^2$ Score สูงสุดที่ **0.9692**
* **การเรียนรู้เชิงลึก (Deep Learning):** สถาปัตยกรรมแบบ **LSTM** สามารถอธิบายความผันแปรของข้อมูลบนเซตทดสอบภาพรวมได้โดดเด่นกว่าด้วยค่า $R^2$ อยู่ที่ **0.9568** ในขณะที่ **GRU** สามารถควบคุมค่าความคลาดเคลื่อนสมบูรณ์เฉลี่ย (MAE) ได้ต่ำที่สุดที่ **1.6121** คะแนน

## โครงสร้างระบบสคริปต์ภายในโปรเจกต์
* `import_data.py`: ระบบสกัดและดึงข้อมูลรายวันผ่านโปรแกรมประยุกต์ Spotify Web API (Data Pipeline)
* `cls_model.py`: สคริปต์การทำความสะอาดฟีเจอร์, ทำ Sliding Window และประเมินผลกลุ่มโมเดลดั้งเดิม
* `rnnv2.py`: โครงสร้างการร้อยเลเยอร์ระบบเครือข่ายจำลองพยากรณ์แบบ Multi-task Learning ของ Keras (TensorFlow)

---
*จัดทำโดย ธีร์ ธรรมโณ คณะวิทยาศาสตร์ ภาควิชาคณิตศาสตร์ สาขาสถิติและวิทยาการข้อมูล มจธ. (KMUTT)*
