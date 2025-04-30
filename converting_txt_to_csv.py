import pandas as pd
def convert_txt_to_csv(txt_path, csv_path):
    data = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                label, text = parts
                label = 'positive' if '__label__2' in label else 'negative'
                data.append([label, text])
    
    df = pd.DataFrame(data, columns=['label', 'text'])
    df.to_csv(csv_path, index=False)

# Convert both
convert_txt_to_csv('train.ft.txt', 'train.csv')
convert_txt_to_csv('test.ft.txt', 'test.csv')
