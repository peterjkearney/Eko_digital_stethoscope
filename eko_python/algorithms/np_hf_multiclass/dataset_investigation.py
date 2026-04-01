import pandas as pd
from pathlib import Path
import os
from datetime import datetime
from tqdm import tqdm

if __name__ == '__main__':
    datapath = Path('/Users/peterkearney/Documents/Eko/data/ext_databases/HF_Lung_V1')
    # BASE_DIR         = Path(__file__).parent
    # EKO_PROJECT_ROOT = BASE_DIR.parent.parent.parent
    # DATA_DIR         = EKO_PROJECT_ROOT / 'data/ext_databases/HF_Lung_V1'

    trainpath = datapath / 'train'
    testpath = datapath / 'test'

    # create manifest file
    # for each txt file, extract event and enter in manifest
    manifest = pd.DataFrame(columns=['Event','StartTime','EndTime','Duration'])

    allTxtFiles = [f.name for f in os.scandir(trainpath) if f.is_file() and f.name.endswith('.txt')]

    for txtFile in tqdm(allTxtFiles,total = len(allTxtFiles),desc = 'Reading text files'):

        filepath = trainpath / txtFile

        fileText = filepath.read_text().splitlines()

        for line in fileText:

            line.strip()

            lineSegs = line.split(' ')

            event = lineSegs[0]
            eventStart = datetime.strptime(lineSegs[1],"%H:%M:%S.%f")
            eventEnd = datetime.strptime(lineSegs[2],"%H:%M:%S.%f")
            eventDuration = pd.Timedelta(eventEnd - eventStart)
            

            manifest_entry = pd.DataFrame([[event, eventStart, eventEnd, eventDuration]],
                columns=['Event', 'StartTime', 'EndTime', 'Duration'])
            
            manifest = pd.concat([manifest, manifest_entry],ignore_index=True)

    manifest.to_csv('manifest.csv',index=False)