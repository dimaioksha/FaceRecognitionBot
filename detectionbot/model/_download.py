import os
from progress.bar import Bar
from google_drive_downloader import GoogleDriveDownloader as gdd


def download() -> None:
    """
    Function to download all required files for model. Applied automaticaly when you create DetectionBot instance.
    """
    id_ = []
    with open('../files/ID.txt', 'r') as f:
        for id in f.readlines():
            id_.append(id.replace('\n', ''))
    paths_ = []
    with open('../files/PATHS.txt', 'r') as f:
        for path in f.readlines():
            paths_.append(path.replace('\n', ''))
    try:
        os.mkdir('../detectionbot/model/age_gender_models')
        os.mkdir('./images')
    except:
        pass
    with Bar('Download', max=len(id_)) as bar:
        for id, path in zip(id_, paths_):
            gdd.download_file_from_google_drive(file_id=id, dest_path=path)
            bar.next()
    print('*******Now Bot is running*******')
