import os
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import uvicorn

app = FastAPI()

# 1. เพิ่ม CORS Middleware เพื่ออนุญาตให้แอปมือถือ (Expo) หรือเว็บอื่นๆ เชื่อมต่อเข้ามาได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตทุกโดเมน (สามารถปรับเป็น IP เฉพาะได้เมื่อใช้งานจริง)
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตทุก HTTP Methods (GET, POST, ฯลฯ)
    allow_headers=["*"],
)

# 2. โหลดไฟล์โมเดล YOLO (ตรวจสอบให้แน่ใจว่าไฟล์ best.pt อยู่ในโฟลเดอร์เดียวกับไฟล์โค้ดนี้)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)


class YOLOPostProcessor:
    """ประมวลผลผลลัพธ์ดิบที่ได้จากโมเดล YOLO ให้เป็น List ของ Predictions"""
    def process(self, results, model_names):
        predictions = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model_names[class_id]
                confidence = float(box.conf[0])

                predictions.append({
                    "food_name_en": class_name,
                    "confidence": round(confidence, 2)
                })
        return predictions


class CaloriesEstimator:
    """แมปชื่อคลาสอาหารภาษาอังกฤษเข้ากับชื่อภาษาไทยและคำนวณปริมาณแคลอรี"""
    NUTRITION_DATABASE = {
        "pork_leg_rice": {"food_name_th": "ข้าวขาหมู", "calories": 690},
        "omelette_rice": {"food_name_th": "ข้าวไข่เจียว", "calories": 450},
        "crispy_pork_basil_rice": {"food_name_th": "ข้าวกะเพราหมูกรอบ", "calories": 750},
        "chicken_rice": {"food_name_th": "ข้าวมันไก่", "calories": 600},
        "red_pork_rice": {"food_name_th": "ข้าวหมูแดง", "calories": 560},
    }

    def estimate(self, predictions):
        enriched_predictions = []
        for pred in predictions:
            food_en = pred.get("food_name_en")
            nutrition = self.NUTRITION_DATABASE.get(food_en, {
                "food_name_th": food_en,
                "calories": 0
            })
            enriched_predictions.append({
                "food_name_en": food_en,
                "food_name_th": nutrition["food_name_th"],
                "confidence": pred.get("confidence", 0.0),
                "calories": nutrition["calories"]
            })
        return enriched_predictions


# สร้าง Instance ของ Processor และ Estimator
post_processor = YOLOPostProcessor()
calories_estimator = CaloriesEstimator()


# Endpoint สำหรับทดสอบว่าเซิร์ฟเวอร์รันสำเร็จหรือไม่ (เมื่อเข้า URL ปกติ)
@app.get("/")
def read_root():
    return {"status": "ok", "message": "FastAPI YOLO Server is running!"}


# Endpoint สำหรับรับรูปภาพจากแอปมือถือมาทำนาย
@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    try:
        # อ่านข้อมูลไฟล์รูปภาพที่ถูกส่งเข้ามา
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # นำรูปไปให้โมเดลทำนาย
        results = model(image)

        # ประมวลผลลัพธ์ด้วย YOLOPostProcessor และ CaloriesEstimator
        raw_predictions = post_processor.process(results, model.names)
        predictions = calories_estimator.estimate(raw_predictions)

        # ส่งผลลัพธ์กลับไปเป็น JSON ให้แอปมือถือแสดงผล
        return {
            "status": "success",
            "filename": file.filename, 
            "predictions": predictions
        }
    except Exception as e:
        # หากมีข้อผิดพลาด (เช่น ส่งไฟล์มาผิดประเภท) จะส่ง Error กลับไปแทนที่จะให้เซิร์ฟเวอร์ล่ม
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # สั่งเปิด API Server ที่พอร์ต 8000 แบบรับการเชื่อมต่อจากทุก IP
    uvicorn.run(app, host="0.0.0.0", port=8000)