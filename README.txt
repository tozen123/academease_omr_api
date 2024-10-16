CNSC ACADMEASE


===========================================================================
Requirements
OpenCV
NumPy
google-cloud-vision
io
os
panda

=============================================================================

FUNCTION : 

def OMR(img, answers, choices=4, is_data=True, widthImg=600, heightImg=600)


Parameters:
	> img: The input image of the answer sheet (as a NumPy array).
	> answers: A list of integers representing the correct answers (0-indexed).
	> choices: An integer specifying the number of choices per question (default is 4).
	> is_data: A boolean indicating whether to return grading data or the final image (default is True).
	> widthImg: The width to which the input image will be resized (default is 600).
	> heightImg: The height to which the input image will be resized (default is 600).

==========================================================================
Returns:
	> If is_data is True, returns a list containing:
		: The identified set number.
		: Recognized digit text.
		: Total score.
		: Percentage rating of the answers.
		: Shaded answers.

	>If is_data is False, returns the final processed image.

==========================================================================
Error Handling

>The OMR function includes error handling to manage common issues such as invalid images or incorrect answer formats. If an error occurs, it will print a message and return None (or an empty list if is_data is True).


