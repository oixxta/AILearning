"""
이미지 디텍션 + TTS

유기동물 사진을 제출하면 탐지 후 안내소로 안내하는 문구를 소리로 표현
TTS(Text to speach) : 텍스트를 소리내서 읽어주는 기능
"""
from gtts import gTTS
#from IPython.display import Audio    # 쥬피터 노트북용 소리 재생
from playsound import playsound      # 로컬 환경용 소리 재생
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime

"""
# TTS 맛보기
def speak_shelter_info(message):
    tts = gTTS(text=message, lang='ko')
    tts.save('yoloText5Sound.mp3')
    #Audio('yoloText5Sound.mp3', autoplay=True)
    playsound('yoloText5Sound.mp3')

myMessage = "고즈넉한 성곽길을 따라 걸을 수 있는 면천읍성에서는 1100"
speak_shelter_info(message=myMessage)
"""

def show_shelter_info_func(region, shelters, detected_info):    #TTS 함수
    shelters_info = shelters.get(region, shelters['Korea'])
    pet_summary = f"{detected_info['count']} animals, {', '.join(detected_info['labels'])} type."
    message = (
        f"Abandoned animal detection results : \n"
        f"Detected animal counts : {detected_info['count']} \n"
        f"Animal type : {', '.join(detected_info['labels'])} \n"
        f"information of {region} Area Animal Protect Center : \n {shelters_info}"
    )
    print('shelter info : ')
    print(message)              #문자 안내

    #음성 안내
    try:
        tts = gTTS(text=f"Abandoned animal in {region} Area has been detected. {pet_summary}. nearest protect shelter : {shelters_info}", lang="en-us")
        tts.save("yoloText5Sound.mp3")
        playsound("yoloText5Sound.mp3")

    except Exception as e:
        print(f'Fail to create voice : {type(e).__name__} - {e}')



def handle_stray_pet_func(region, shelters, detected_info):
    print('Presumed to be an abandoned animal.')
    show_shelter_info_func(region, shelters, detected_info)

"""
#테스트용 셈플
region = "Gangnam"
shelters = {            #보호소 정보
    "Seoul" : "Seoul Animal Protect Center : 02-1234-5678",
    "Korea" : "National Federation of Animal Protect : 1577-1000"
}
detected_info = {
    "count":3,
    "labels":['Tiger', 'Lion', 'Elephant']
}
handle_stray_pet_func(region, shelters, detected_info)
"""


# 탐지정보를 로그 파일로 저장
def save_detection_log_func(image_path, detection_data):
    log_file_name = 'yotest5detec.txt'
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(log_file_name, 'a', encoding='utf-8') as f:
        f.write(f'\n[{now}] image : {image_path}\n')
        f.write(f'detected object counts : {len(detection_data)}\n')
        for d in detection_data:
            f.write(f" - {d['label']} : box = {d['box']}, confidence = {d['confidence']:.2f} \n")
        f.write("-" * 40 + "\n")
    print(f'detection result saved in {log_file_name}.')

# 유기동물 감지 함수
def detect_pets_func(image_path):
    pet_desc = {
        'dog' : 'puppy',
        'cat' : 'kitty',
    }
    shelters = {
        "Seoul" : "Seoul Animal Protect Center : 02-1234-5678",
        "Busan" : "Busan Animal Protect Center : 051-9999-7878",
        "Korea" : "National Federation of Animal Protects : 1577-1000"
    }
    stray_keywords = ['street', 'road', 'outside', 'stray']
    model = YOLO('yolov8m.pt')
    image = cv2.imread(image_path)
    if image is None :
        print("we can't check your image")
        return
    results = model(image)
    detected_animals = []
    detection_data = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()

            if label in pet_desc:
                detected_animals.append(label)
                detection_data.append(
                    {
                        'label' : pet_desc[label],
                        'box' : (x1, y1, x2, y2),
                        'confidence' : confidence
                    }
                )
                cv2.rectangle(image, (x1,y1),(x2,y2), (0, 255, 0), 2)
                cv2.putText(image, f'{label} : {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #결과 이미지 저장
    output_path = 'outtest5.jpg'
    cv2.imwrite(output_path, image)
    print('detected image has been save scucessful')

    #이미지 보기
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    #소리 출력
    if detected_animals:    # detected_animals 리스트에 데이터가 있을 경우 True, 없으면 False
        print('Detected Animal : ')
        for pet in set(detected_animals):
            print(f'- {pet_desc.get(pet, pet)}')
        
        # 감지된 정보들을 텍스트 파일로 저장
        save_detection_log_func(image_path, detection_data)

        # 유기동물 조건을 확인해서(이미지 경로에 'street', 'road', 'outside', 'stray' 키워드가 있다면)
        if any(pet in ['dog', 'cat'] for pet in detected_animals) and any(keyword in image_path.lower() for keyword in stray_keywords):
            detected_info = {
                'count' : len(detection_data),
                'labels' : sorted(set([d['label'] for d in detection_data]))
            }
            handle_stray_pet_func(region='Korea', shelters=shelters, detected_info=detected_info)
    else:
        print('Not detected animal')


detect_pets_func('street_cat.jpg')