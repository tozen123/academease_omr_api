import cv2
import numpy as np
import utils  
import io

"""





"""

def OMR(img, answers, choices = 4, is_data=True, widthImg = 600, heightImg = 600, bw_ratio = 60):
    try:
        # Check if img and answers are valid
        if img is None:
            raise ValueError("Input image is invalid.")
        if not isinstance(answers, list) or not all(isinstance(ans, int) for ans in answers):
            raise ValueError("Answers must be a list of integers.")
        
        # widthImg = 600
        # heightImg = 600
        # choices = 4

        # Reshape answers for multiple sets
        ans_final = [answers[i:i + 10] for i in range(0, len(answers), 10)]
        questions = len(ans_final[0])

        #set  flag
        is_imgQuality = True

        # Convert to grayscale and preprocess image
        img = cv2.resize(img, (widthImg, heightImg))
        imgContours = img.copy()
        imgsetContours = img.copy()
        imgDigit = img.copy()
        imgFinal = img.copy()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, 10, 50)

        # Find all contours
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

        # Find rectangles (assumed to be function in utils.py)
        rectCon = utils.rectContour(contours)
        print(len(rectCon))

        base = len(rectCon) -4


        if len(rectCon) < 2:
            raise ValueError("Not enough contours detected for the answer sheet.")



        # Digit Recognition
        digitContour = utils.getCornerPoints(rectCon[4])
        digitContour = utils.reorder(digitContour)

        # Call digit_recognition with the image and reordered points
        digit_text = utils.digit_recognition(img, digitContour)



        # FOR SET IDENTIFICATION
        setContour = utils.getCornerPoints(rectCon[5])
        setVal = utils.determine_set(imgsetContours, setContour, widthImg, heightImg)

        points_origin = {}
        points = []

        # Process each detected contour
        for i in range(0, len(rectCon) - base):
            a = utils.getCornerPoints(rectCon[i])
            b = utils.reorder(a)[0][0]  

            points_origin[i] = [b[0], b[1]]
            points.append((b[0], b[1]))

        # Sort points and match them
        points_sorted_by_y = sorted(points, key=lambda x: x[1])
        point_4 = points_sorted_by_y[-1]
        points_remaining_sorted = sorted(points_sorted_by_y[:-1], key=lambda x: x[0])

        final_points = points_remaining_sorted + [point_4]
        keys_found = []

        for _, pts in enumerate(final_points):
            if isinstance(pts, (list, tuple)) and len(pts) >= 2:
                for key, value in points_origin.items():
                    if list(pts) == value:
                        keys_found.append(key)
                        break

        scores = []
        shaded_answer = []

        # Process each scene (group of questions)
        for scene in range(len(keys_found)):
            img_current = img if scene == 0 else imgFinal
            biggestContour = utils.getCornerPoints(rectCon[keys_found[scene]])
            ans = ans_final[scene]

            cv2.drawContours(imgContours, biggestContour, -1, (0, 255, 0), 10)
            biggestContour = utils.reorder(biggestContour)

            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])

            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpColored = cv2.warpPerspective(img_current, matrix, (widthImg, heightImg))

            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]

            total_white_pixels, total_black_pixels = utils.count_white_black_pixels(imgThresh)
            rat = (total_white_pixels / (total_black_pixels + total_white_pixels) )*100
            print("BW Ratio  : ", rat)
            #is_imgQuality = True
            
            if rat >= bw_ratio:
                is_imgQuality = False
                break
                            

            boxes = utils.splitBoxes(imgThresh, questions, choices)

            # Get the pixel values for each question box
            myPixelVal = np.zeros((questions, choices))
            countC = 0
            countR = 0
            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if countC == choices:
                    countR += 1
                    countC = 0

            # Determine shaded answers
            myIndex = []
            for x in range(0, questions):
                arr = myPixelVal[x]
                mean = sum(arr) / len(arr)

                # if max(arr) > (mean * 1.5):
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])

                # else:
                #     myIndex.append(4)  # Invalid marking

            shaded_answer.append(myIndex)

            # Grading
            grading = []
            for x in range(0, questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            scores.append(sum(grading))

            # Visualization
            imgResult = imgWarpColored.copy()
            imgResult = utils.showAnswers(imgResult, myIndex, grading, ans, questions, choices)

            imgRawDrawing = np.zeros_like(imgWarpColored)
            imgRawDrawing = utils.showAnswers(imgRawDrawing, myIndex, grading, ans, questions, choices)

            invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWarp = cv2.warpPerspective(imgRawDrawing, invMatrix, (widthImg, heightImg))

            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)

            # Return data based on is_data flag

     
        if is_data and is_imgQuality:
            score = sum(scores)
            rating = (score/len(answers))*100
            rating = round(rating, 4)
            rating = str(rating) + " %"
            return [setVal,digit_text, score, rating, shaded_answer]
        elif not is_data and is_imgQuality:
            return imgFinal
        else:
            return "Poor Image Quality"

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None if not is_data else []




# if __name__ == "__main__":
#     img = cv2.imread('D:/~new_api_omr/uploads/JPEG_20241001_145101_7245154632240040578.jpg')
#     answers = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3]


#     # Data 
#     a = OMR(img, answers)
#     print(a)
    
    
#     b = OMR(img, answers, is_data = False)
#     cv2.imshow("Final Image", b)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()