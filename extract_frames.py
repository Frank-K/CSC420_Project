import cv2
vidcap = cv2.VideoCapture('Avengers_Endgame_World_Premiere_Dazzles_With_Epic_and_Emotional_Ending.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
  if count >= 1000:
    success = False