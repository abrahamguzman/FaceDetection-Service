from deepface import DeepFace
import cv2

img1 = cv2.imread('./child03.jpg')
#plt.imshow(img1[:,:,::-1])
#plt.show()
result = DeepFace.analyze(img1, actions=['emotion'])
print(result)
print('*****************************')
#emotion = result

# Imprimir la emoción dominante

print('Emocion dominante:', result[0]['dominant_emotion'])