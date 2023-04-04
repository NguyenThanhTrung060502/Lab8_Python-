import cv2 
import numpy as np 

# Request_1

def request_1():
    image = cv2.imread('variant_3.png')
    image = cv2.resize(image, (900,600))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    cv2.imshow("HSV", hsv)
    cv2.waitKey()
    cv2.destroyAllWindows()


# Request 2 and request 3

def requests_2_3():

    # Создание объекта VideoCapture для захвата видео с камеры 
    cap = cv2.VideoCapture(0) 
    
    # Цикл обработки каждого кадра видео 
    while True: 
        # Получение кадра видео с камеры 
        ret, frame = cap.read() 
        if not ret: 
            break 
    
        # Преобразование цветных фотографий в черно-белые
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
        ret, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY) 
        # Найти границу
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
        img_copy = frame.copy() 

        # frame.shape[0] = ось х and frame.shape[1] = ось y
    
        # Нарисовать прямоугольник размером 200x200
        cv2.rectangle(img_copy, (frame.shape[1] // 2 - 100, frame.shape[0] // 2 - 100), 
                    (frame.shape[1] // 2 + 100, frame.shape[0] // 2 + 100), (0, 0, 255), 2) 
    
        for i, contour in enumerate(contours):  # Цикл по одной области контура
            color = (0, 255, 0) 
            for j, contour_point in enumerate(contour):  # Цикл по точкам

                # contour_point[0] = (x_i, y_i) -> x_i = contour_point[0][0] and y_i = contour_point[0][1] 

                if contour_point[0][1] >= frame.shape[0] / 2 - 100 \
                    and contour_point[0][1] <= frame.shape[0] / 2 + 100 \
                    and contour_point[0][0] >= frame.shape[1] / 2 - 100 \
                    and contour_point[0][0] <= frame.shape[1] / 2 + 100: 
                        
                    color = (0, 0, 255) 
    
            for j, contour_point in enumerate(contour): 
                cv2.circle(img_copy, ((contour_point[0][0],  contour_point[0][1])), 2, color, 2, cv2.LINE_AA) 
    
        cv2.imshow('Request 2 and Request 3', img_copy) 
    
        # Выход из цикла по нажатию клавиши "q" 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
    
    # Освобождение ресурсов 
    cap.release() 
    cv2.destroyAllWindows()


# Additional request

def additional():
        
    # Создание объекта VideoCapture для захвата видео с камеры 
    cap = cv2.VideoCapture(0) 
    
    # Цикл обработки каждого кадра видео 
    while True: 
        # Получение кадра видео с камеры 
        ret, frame = cap.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA) 
        if not ret: 
            break 
    
        # Преобразование цветных фотографий в черно-белые 
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
        ret, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY) 
        # Найдите границу 
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_NONE) 
    
        img_copy = frame.copy() 
        fly = cv2.imread("fly64.png", cv2.IMREAD_UNCHANGED) 
        fly = cv2.resize(fly, (8, 8)) 
    
        for i, contour in enumerate(contours):  # Цикл по одной области контура
            for j, contour_point in enumerate(contour):  # Цикл по точкам
                x, y = contour_point[0] 

                if y - fly.shape[0] // 2 >= 0 and y + fly.shape[0] // 2 < img_copy.shape[0] \
                    and x - fly.shape[1] // 2 >= 0 and x + fly.shape[1] // 2 < img_copy.shape[1]: 

                    if (np.min(img_copy[y - fly.shape[0] // 2:y + fly.shape[0] // 2, 
                                        x - fly.shape[1] // 2:x + fly.shape[1] // 2, 3]) == 0):  
                        continue 

                    img_copy[y - fly.shape[0] // 2:y + fly.shape[0] // 2, 
                            x - fly.shape[1] // 2:x + fly.shape[1] // 2, :3] = fly[:, :, :3]  

                    # Отметьте точки, в которые было вставлено изображение мухи.
                    img_copy[y - fly.shape[0] // 2:y + fly.shape[0] // 2, 
                            x - fly.shape[1] // 2:x + fly.shape[1] // 2, 3] = 0 
    
        img_copy = img_copy[:, :, :3] 
        cv2.imshow('Additional request', img_copy) 
    
        # Выход из цикла по нажатию клавиши "q" 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
    
    # Освобождение ресурсов 
    cap.release() 
    cv2.destroyAllWindows()


request_1()
requests_2_3()
additional()