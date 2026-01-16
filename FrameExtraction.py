import cv2
import matplotlib.pyplot as plt
import IPython.display as ipd
import csv

plt.style.use('ggplot')


def showFrames(vidDir,frameStart):
    '''
    Shows a 5x5 grid of 25 frames in a video.

    Args:
        vidDir: The directory(absolute/relative) an mp4 video and 
        framStart: Desired frame to start

    Returns:
        Nothing.
    '''
    ipd.Video(vidDir, width=700)
    vid = cv2.VideoCapture(vidDir)
    print("Opened: ", vid.isOpened())
    
    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frameCount)

    videoHeight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    videoWidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    print("Height:", videoHeight, "\nWidth:", videoWidth)

    figure, frameGrid = plt.subplots(5,5, figsize=(13,10))
    frameGrid = frameGrid.flatten()

    imgIdx = 0
    frameStop = frameStart+25
    for frameNum in range(frameCount):
        ret,frame = vid.read()
        if ret == False:
            print("Success")
            break
        if frameNum >= frameStart:
            if frameNum>=frameStop:
                break
            frameGrid[imgIdx].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frameGrid[imgIdx].set_title(f'Frame: {frameNum}')
            frameGrid[imgIdx].axis('off')
            imgIdx += 1
    vid.release()
    plt.tight_layout()
    plt.show()

#Helper func to write into csv file (Create data set)
def write(imageName, state):
    data = [imageName, state]
    with open('DataSet.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


'''
Takes in the directory of mp4 video and the output directory along with desired frame to start and the video name
Outputs 25 frames starting from start frame as png into the output directory
'''
def extractFrames(vidDir,outDir,frameStart,name, writeCsv):
    '''
    Extracts 25 frames into a given directory and based on name writes into a csv file

    Args:
        vidDir: Directory of the video
        outDir: Directory of where the images should be extracted
        frameStart: The desired frame to start
        name: The names of the video (will also be in name of images when extracted)
        writeCsv: True if the images should be written to the csv file "DataSet.csv"
    Returns:
        Nothing.

    '''
    ipd.Video(vidDir, width=700)
    vid = cv2.VideoCapture(vidDir)
    print("Opened: ", vid.isOpened())
    
    frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frameCount)

    videoHeight = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    videoWidth = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    frameStop = frameStart+25
    for frameNum in range(frameCount):
        ret,frame = vid.read()

        if ret == False:
            print("Success")
            break

        if frameNum >= frameStart:
            if frameNum>=frameStop:
                break
            fileName = name + r'_' + repr(frameNum)+ r'.png'
            if writeCsv == True:
                    match name[:6]:
                        case 'Geeked':
                            write(fileName, 1)
                        case 'Locked':
                            write(fileName, 0)
            fileName = '\\' + fileName
            cv2.imwrite(outDir+fileName, frame)
    vid.release()

vid1 = 'Geeked1.mp4'
vid2 = 'Geeked2.mp4'
vid3 = 'Geeked3.mp4'

locked1 = 'Locked1.mp4'
locked2 = 'Locked2.mp4'

geekedFolder = 'GeekedFolder'
lockedFolder = 'LockedFolder'

if __name__ == "__main__":
    # Example usage:
    # showFrames(vid1, frameStart=1)
    # extractFrames(vid1, "Images", frameStart=1, name="Geeked1", writeCsv=True)
    pass
