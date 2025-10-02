# 자료구조 중 queue : FIFO 구조
from collections import deque

class MyQueue:
    def __init__(self, iterable=None):
        # queue는 양쪽 끝에서 삽입 / 삭제가 이루어짐 ( O(1) )
        self._data = deque()

        if iterable is not None:
            for x in iterable:
                self.enqueue(x)
    
    def enqueue(self, x):
        self._data.append(x)    # 뒤(rear, back)에 원소를 추가함.
        return x
    
    def dequeue(self):      # 앞(front)의 원소 제거
        if not self._data:
            raise IndexError('dequeue from empty queue')
        return self._data.popleft()

    def front(self):        # 큐에서 맨앞 원소 확인용 매소드
        if not self._data:
            raise IndexError('front from empty queue')
        return self._data[0]
    
    def is_empty(self):
        return not self._data
    
    def check_size(self):
        return len(self._data)

    def clear(self):
        self._data.clear()
    
    def __repr__(self):     #객체를 문자열로 표현할 때 사용
        return f'Queue (front -> back {list(self._data)})'

def demo_fifo():
    q = MyQueue()
    for item in ['a', 'b', 'c', 'd']:
        q.enqueue(item)
        print(f'enque {item} -> ', q)
    print('\n Dequeue until empty (FIFO) ')

    while not q.is_empty():
        print(f'deenque -> ', q.dequeue(), ' | now:', q)
    
demo_fifo()


# 활용 1: 큐(Queue) 자료구조를 이용해서 프린터 작업 처리 시뮬레이션을 구현

def simulate_printer(jobs, ppm=15):
    # jobs: [(문서이름, 페이지수), ...],   jobs는 출력할 문서와 페이지 수의 리스트
    # 예: [("report.pdf", 10), ("slides.pptx", 30), ("invoice.docx", 5)]
    # ppm(pages per minute) : 분당 몇 페이지 인쇄 가능한지
    q = MyQueue(jobs)   # FIFO 구조
    t_sec = 0.0           # 시뮬레이션 시간 누적
    order = []            # 실제 처리된 문서 순서를 저장할 리스트

    print("\n[Printer Queue Test]")
    while not q.is_empty():
        doc, pages = q.dequeue()           #  맨 앞의 작업 꺼내기
        duration = (pages / ppm) * 60.0  # 초 단위 처리시간 - 걸리는 시간(초)
        t_sec += duration            
        order.append(doc)                     # 처리 순서 기록
        print(f"t={t_sec:6.1f}s | printed: {doc:10s} ({pages}p)")

    print("처리 순서(FIFO):", order)     # 모든 문서 출력이 끝나면 처리 순서를 보여줌

# 프린터 작업 큐
jobs = [("report.pdf", 10), ("slides.pptx", 30), ("invoice.docx", 5)]
simulate_printer(jobs, ppm=20)


# 활용 2: TensorFlow의 tf.data.Dataset.prefetch()가 내부적으로 Queue와 비슷하게 동작한다는 것을 보여주기 위한 비교 예.
# TensorFlow prefetch
# ds = ds.prefetch(2) 는 데이터 소비자(모델 학습 루프) 가 데이터를 꺼내 쓰기 전에 프로듀서(데이터 준비 과정) 가 2개를 미리 준비해두는 버퍼 역할.
# 즉, 데이터가 FIFO 큐에 미리 들어가 있다가 모델이 필요할 때 꺼내오는 구조.
# 실행해보면 첫 번째 원소는 1초 지연이 있지만, 이후 원소는 지연 없이 바로 출력. (큐에 미리 쌓여 있었기 때문)

import tensorflow as tf 
import time 
from queue import Queue 
import threading   # 병렬 실행용 스레드 모듈

# 1) TensorFlow Dataset + prefetch
def tf_prefetch_demo():
    ds = tf.data.Dataset.range(1, 6)    # 1~5까지 숫자를 원소로 하는 Dataset 생성

    # 각 원소를 가져올 때 1초 대기 -> 데이터 준비 시간이 걸리는 상황을 흉내냄
    ds = ds.map(lambda x: tf.py_function(lambda y: (time.sleep(1), y)[1], [x], tf.int64))  

    ds = ds.prefetch(2)  # Prefetch 버퍼 크기 = 2 → 2개를 미리 준비해 둠

    print("TensorFlow Prefetch Demo") #  실행 구분 출력
    start = time.time()  
    for x in ds:     # Dataset 순회 (소비자 역할)
        print("받은 데이터:", x.numpy(), f"(elapsed {time.time()-start:.2f}s)")  # 데이터 소비 시점과 경과 시간 출력
    print()

# 2) Python Queue 로 동작 재현
def producer(q: Queue):        # 생산자 스레드 함수
    for i in range(1, 6):
        time.sleep(1)               # 데이터 준비에 1초 소요했다고 가정
        q.put(i)                      # 큐에 데이터 넣기 (enqueue)
        print(f"[Producer] {i} 준비")    # 생산자 로그 출력

def consumer(q: Queue):      # 소비자 스레드 함수
    start = time.time()      
    for _ in range(5):             # 5번 데이터 소비
        item = q.get()             # 큐에서 데이터 꺼내기 (dequeue, FIFO)
        print("받은 데이터:", item, f"(elapsed {time.time()-start:.2f}s)")   # 소비된 데이터와 경과 시간 출력
        q.task_done()       

def python_queue_demo():
    q = Queue(maxsize=2)   # 큐 크기 제한 = 2 (prefetch(2)와 유사)
    t1 = threading.Thread(target=producer, args=(q,))   # 생산자 스레드 생성
    t2 = threading.Thread(target=consumer, args=(q,))  # 소비자 스레드 생성
    t1.start(); t2.start()      # 스레드 실행 시작
    t1.join(); t2.join()        # 두개의 스레드가 모두 끝날 때까지 메인 스레드는 대기
    print() 

# 실행
tf_prefetch_demo()    # TensorFlow Prefetch 처리
python_queue_demo()   # Python Queue 처리

    