import boss_input as BI

def dataset_making(InputCapture,OutputFile):
    BI.CaptureFace(InputCapture,OutputFile)

if __name__ == '__main__':
    dataset_making('./Capture/Text1.mp4','./data/2/')