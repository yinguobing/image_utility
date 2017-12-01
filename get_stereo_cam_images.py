import cv2


cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)

count = 0

while True:

    result1, frame1 = cap1.read()
    result2, frame2 = cap2.read()

    cv2.imshow("left", frame1)
    cv2.imshow("right", frame2)

    key_pressed = cv2.waitKey(30)
    if key_pressed == 27:
        break
    elif key_pressed == ord('s'):
        count += 1
        if count <= 9:
            left_name = "left0"+str(count)+".jpg"
            right_name = "right0"+str(count)+".jpg"
        else:
            left_name = "left"+str(count)+".jpg"
            right_name = "right"+str(count)+".jpg"
        cv2.imwrite(left_name, frame1)
        cv2.imwrite(right_name, frame2)

cap1.release()
cap2.release()
cv2.destroyAllWindows()
