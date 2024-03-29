import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)

user_dict = {"Landon":"landon.jpg", "Lorna":"lorna.jpg"}

known_face_encodings, known_face_names = [], []

for user in user_dict:
	# load and encode picture of user
	curImage = face_recognition.load_image_file(user_dict[user])
	face_encoding = face_recognition.face_encodings(curImage)[0]

	# add encoding and picture to respective lists at same index
	known_face_encodings.append(face_encoding)
	known_face_names.append(user)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
	# Grab a single frame of video
	ret, frame = video_capture.read()

	# Only process every other frame of video to save time
	if process_this_frame:
		# Resize frame of video to 1/4 size for faster face recognition processing
		small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		rgb_small_frame = small_frame[:, :, ::-1]

		# Find all the faces and face encodings in the current frame of video
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		face_names = []
		for face_encoding in face_encodings:
			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
			name = "Unknown"

			# If a match was found in known_face_encodings, just use the first one.
			if True in matches:
				first_match_index = matches.index(True)
				name = known_face_names[first_match_index]

			face_names.append(name)

	process_this_frame = not process_this_frame

	if len(face_names) == 2:

		locs1 = face_locations[0]
		locs2 = face_locations[1]

		top1 = 4 * (locs1[0])
		right1 = 4 * (locs1[1])
		bottom1 = 4 * (locs1[2])
		left1 = 4 * (locs1[3])

		top2 = 4 * (locs2[0])
		right2 = 4 * (locs2[1])
		bottom2 = 4 * (locs2[2])
		left2 = 4 * (locs2[3])

		# swap face 1 for face 2
		yAxis1 = bottom1 - top1
		xAxis1 = right1 - left1

		yAxis2 = bottom2 - top2
		xAxis2 = right2 - left2

		face1_array = np.copy(frame[top1:bottom1, left1:right1])
		face2_array = np.copy(frame[top2:bottom2, left2:right2])

		# impose smaller face onto larger face
		if xAxis1 < xAxis2 and yAxis1 < yAxis2:
			# potential bug but attempted to represent 50% of each face on faceq
			frame[top1:bottom1, left1:right1] = (face2_array)[0:yAxis1, 0:xAxis1]
		elif xAxis1 > xAxis2 and yAxis1 > yAxis2:
			frame[top2:bottom2, left2:right2] = (face1_array)[0:yAxis2, 0:xAxis2]
		else:
			pass

	for (top, right, bottom, left), name in zip(face_locations, face_names):
		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

	# Display the resulting image
	cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()


# def single_face():
# 	if len(face_names) == 1:
# 		locs1 = face_locations[0]
#
# 		top1 = 4* (locs1[0])
# 		right1 = 4* (locs1[1])
# 		bottom1 = 4* (locs1[2])
# 		left1 = 4* (locs1[3])
#
# 		face1 = frame[top1:bottom1, left1:right1]
#
# 		yAxis1 = bottom1 - top1
# 		xAxis1 = right1 - left1
# 		print(f"y: {yAxis1}\nx: {xAxis1}")
#
# 		face2 = np.copy(face1[0:yAxis1, 0:xAxis1])
# 		plt.imshow(face2)
# 		plt.show()
# 	break
#
