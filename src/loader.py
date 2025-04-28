import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import os
from PIL import Image

def get_prefix_split_helper(s):
    return s.split('-')[0]

def get_second_split_helper(s):
    return s.split('-')[1]

def collate_fn(batch):
    # Get the maximum number of text lines in this batch
    max_text_lines = max(len(item['texts']) for item in batch)
    
    # Initialize lists for batched data
    writer_ids = []
    images = []
    xml_paths = []
    texts = []
    num_text_lines = []
    
    # Fill in the lists
    for item in batch:
        writer_ids.append(item['writer_id'])
        images.extend(item['images'])
        xml_paths.extend(item['xml_paths'])
        texts.extend(item['texts'])
        num_text_lines.append(len(item['texts']))
    
    return {
        'writer_id': writer_ids,
        'images': images,
        'xml_paths': xml_paths,
        'texts': texts,
        'num_text_lines': num_text_lines
    }

class IAMDataset(Dataset):
    def __init__(self, img_path, xml_path, line_path, transform=None):
        # Assert that all paths exist
        if not os.path.exists(img_path):
            raise ValueError(f"Image path does not exist: {img_path}")
        if not os.path.exists(xml_path):
            raise ValueError(f"XML path does not exist: {xml_path}")
        if not os.path.exists(line_path):
            raise ValueError(f"Line path does not exist: {line_path}")
            
        self.img_path = img_path
        self.xml_path = xml_path
        self.line_path = line_path
        self.transform = transform # TODO: Add transform. I think chiroDiff does some rescaling but not sure about One-DM

        # load data
        self.data = self.load_data()
        
    def load_data(self):
        data = []
        for root, dirs, files in os.walk(self.xml_path):
            for file in files:
                if file.endswith('.xml'):
                    xml_file = os.path.join(root, file)
                    xml_data = self.load_single_data(xml_file)
                    if xml_data is not None:
                        data.append(xml_data)
        return data
    
    def load_single_data(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find the Form element and get the writerID
        form_element = root.find('.//Form')
        if form_element is not None:
            writer_id = form_element.get('writerID')
            
            # Find all TextLine elements
            text_lines = root.findall('.//TextLine')
            text_line_data = []
            for text_line in text_lines:
                id = text_line.get('id')
                first_split = get_prefix_split_helper(id)
                second_split = get_second_split_helper(id)[:-1]
                text_line_data.append({
                    'id': id,
                    'img_path': os.path.join(self.img_path, 
                                             first_split + 
                                             "/" + first_split + "-" + second_split + 
                                             "/" + id + '.png'),
                    'xml_path': os.path.join(self.line_path, 
                                             first_split + 
                                             "/" + first_split + "-" + second_split + 
                                             "/" + id + '.xml'),
                    'text': text_line.get('text')
                })
            
            return {
                'writer_id': writer_id,
                'text_lines': text_line_data,
            }
        
        return None
        
    def __len__(self):
        return len(self.data)
      
    def __getitem__(self, index):
        # Get a single sample
        sample = self.data[index]
        writer_id = sample['writer_id']
        
        # Load images for each text line
        images = []
        xml_paths = []
        texts = []
        for text_line in sample['text_lines']:
            # Construct the image path using the text line ID
            img_path = text_line['img_path']
            
            # Load and convert the image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)

            xml_paths.append(text_line['xml_path'])
            texts.append(text_line['text'])
            
        return {
            'writer_id': writer_id,
            'images': images,
            'xml_paths': xml_paths,
            'texts': texts
        }

# Then use the standard DataLoader with your custom Dataset
# dataset = IAMDataset('./data/lineImages', './data/original', './data/lineStrokes')
# dataloader = DataLoader(
#     dataset,
#     batch_size=32,
#     shuffle=True,
#     num_workers=4
# )