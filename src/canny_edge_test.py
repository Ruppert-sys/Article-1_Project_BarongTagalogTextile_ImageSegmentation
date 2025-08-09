import cv2

def apply_canny_edge(frame_):
	gray = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edges_ = cv2.Canny(blurred, 50, 150)
	return edges_


cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	
	if not ret:
		break
	
	edges = apply_canny_edge(frame)
	cv2.imshow('Original', frame)
	cv2.imshow('Edges', edges)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
