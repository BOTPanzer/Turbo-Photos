import json
from PIL import Image
import numpy as np
import cv2
from os import listdir, rename
from os.path import isdir, isfile, join, getmtime
import sys



# Util
class Util:
  @staticmethod
  def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
      json.dump(data, f, ensure_ascii=False) # Uglyer but faster and smaller size
      #json.dump(data, f, ensure_ascii=False, indent=2)

  @staticmethod
  def load_json(path):
    try:
      with open(path) as f:
        return json.load(f)
    except:
      return {}
  
  @staticmethod
  def load_images(images_folder, metadata, metadata_list, no_metadata_list):
    # Get images sorted by modified date
    paths = listdir(images_folder)

    # Loop images
    for image_name in paths:
      # Not a valid format
      valid = False
      image_name_lower = image_name.lower()
      for format in ['.png', '.jpg', '.jpeg', '.webp']:
        if image_name_lower.endswith(format):
          valid = True
          break
      if not valid: continue

      # Not a file
      if not isfile(join(images_folder, image_name)): continue

      # Check if file has metadata
      if image_name in metadata:
        metadata_list.append(image_name)
      else:
        no_metadata_list.append(image_name)
    
    # Debugging
    print('Images with metadata: ' + str(len(metadata_list)))
    print('Images without metadata: ' + str(len(no_metadata_list)))

# AI Models
class DescriptionModel:
  device = None
  torch_dtype = None
  model = None
  processor = None

  def run(self, image, prompt):
    # No model -> Load it
    if (self.model == None):
      # Import LLM libraries
      print('Importing LLM libraries...')
      import torch
      from transformers import AutoProcessor, AutoModelForCausalLM

      # Select device
      print('Loading LLM for the first time... (CUDA = ' + str(torch.cuda.is_available()) + ')')
      self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
      self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

      # Load model
      model_path = './Florence2'
      self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
      self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Run prompt
    inputs = self.processor(text=prompt, images=image, return_tensors='pt').to(self.device, self.torch_dtype)

    generated_ids = self.model.generate(
      input_ids=inputs['input_ids'],
      pixel_values=inputs['pixel_values'],
      max_new_tokens=1024,
      num_beams=3
    )

    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = self.processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
    return parsed_answer

class TextDetectionModel:
  ocr = None

  def run(self, image):
    # No model -> Load it
    if (self.ocr == None):
      # Disable PaddleOCR logging
      import logging
      logging.getLogger('ppocr').setLevel(logging.ERROR)

      # Import LLM libraries
      print('Importing PaddleOCR libraries...')
      from paddleocr import PaddleOCR
      self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

    # Look for text in the image
    numpy_image = np.array(image.convert('RGB'))
    numpy_image_cv2 = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) #convert to BGR
    result = self.ocr.ocr(numpy_image_cv2, cls=True)

    return result



#   /$$$$$$  /$$           /$$                 /$$
#  /$$__  $$| $$          | $$                | $$
# | $$  \__/| $$  /$$$$$$ | $$$$$$$   /$$$$$$ | $$
# | $$ /$$$$| $$ /$$__  $$| $$__  $$ |____  $$| $$
# | $$|_  $$| $$| $$  \ $$| $$  \ $$  /$$$$$$$| $$
# | $$  \ $$| $$| $$  | $$| $$  | $$ /$$__  $$| $$
# |  $$$$$$/| $$|  $$$$$$/| $$$$$$$/|  $$$$$$$| $$
#  \______/ |__/ \______/ |_______/  \_______/|__/

# Model
model_description = DescriptionModel()
model_text = TextDetectionModel()

# Options
images_folder = ''
metadata_file = ''
save_every = 5  # When creating or fixing metadata, progress gets saved every this amount of completed files. Only the last save sorts by last mofified (faster this way)

# Metadata
metadata = {}

# Images lists
images_with_metadata = []
images_without_metadata = []



#  /$$      /$$             /$$                     /$$             /$$              
# | $$$    /$$$            | $$                    | $$            | $$              
# | $$$$  /$$$$  /$$$$$$  /$$$$$$    /$$$$$$   /$$$$$$$  /$$$$$$  /$$$$$$    /$$$$$$ 
# | $$ $$/$$ $$ /$$__  $$|_  $$_/   |____  $$ /$$__  $$ |____  $$|_  $$_/   |____  $$
# | $$  $$$| $$| $$$$$$$$  | $$      /$$$$$$$| $$  | $$  /$$$$$$$  | $$      /$$$$$$$
# | $$\  $ | $$| $$_____/  | $$ /$$ /$$__  $$| $$  | $$ /$$__  $$  | $$ /$$ /$$__  $$
# | $$ \/  | $$|  $$$$$$$  |  $$$$/|  $$$$$$$|  $$$$$$$|  $$$$$$$  |  $$$$/|  $$$$$$$
# |__/     |__/ \_______/   \___/   \_______/ \_______/ \_______/   \___/   \_______/

# Images & metadata (per file)
def metadata_needs_fix(metadata):
  return 'caption' not in metadata or 'labels' not in metadata or 'text' not in metadata

def fix_image_metadata(image, metadata):
  # Caption missing
  if 'caption' not in metadata:
    print('Generating caption (very detailed)...')
    prompt = '<MORE_DETAILED_CAPTION>' # <CAPTION> <DETAILED_CAPTION>
    metadata['caption'] = model_description.run(image, prompt)[prompt].strip()

  # Labels missing
  if 'labels' not in metadata:
    print('Generating labels...')
    prompt = '<OD>'
    metadata['labels'] = model_description.run(image, prompt)[prompt]['labels']

  # Text missing
  if 'text' not in metadata:
    print('Generating text scan...')
    ocr_text = model_text.run(image)
    texts = []
    for ocr in ocr_text:
      if ocr == None: continue
      for region in ocr:
        textAndConfidence = region[1] # [0] = text, [1] = confidence
        text = textAndConfidence[0].strip()
        confidence = textAndConfidence[1]
        if confidence > 0.5:
          texts.append(text)

    metadata['text'] = texts

# Images & their metadata (lists)
def search():
  # Ask for search
  search = ''
  while search == '':
    search = input('\nSearch: ')
  search = search.lower()

  # Search in metadata files
  for i in range(0, len(images_with_metadata)):
    # Get image & load its metadata
    image_path = join(images_folder, images_with_metadata[i])
    image_metadata = metadata[images_with_metadata[i]]

    # Check if metadata contains search
    if 'caption' in image_metadata and search in image_metadata['caption'].lower(): 
      print(image_path)
    elif 'labels' in image_metadata and search in image_metadata['labels']:
      print(image_path)

def create_all_metadata():
  # Temp vars
  isFirstSave = True
  created = 0

  # Loop images without metadata
  size = len(images_without_metadata)
  for i in reversed(range(0, size)):
    # Get image name & path
    image_name = images_without_metadata[i]
    image_path = join(images_folder, image_name)

    # Create empty metadata
    image_metadata = {}

    # Load image
    image_number = size - i
    print('\nCreating (' + str(image_number) + ' of ' + str(size) + '): ' + image_path)
    image = Image.open(image_path)

    # Fix metadata (generate missing keys, aka all)
    fix_image_metadata(image, image_metadata)

    # Move image from a list to another
    images_with_metadata.append(image_name)
    images_without_metadata.pop(i)

    # Save image metadata
    metadata[image_name] = image_metadata
    created += 1

    # Save metadata to prevent losing progress (without sorting, is faster)
    if image_number != size and created % save_every == 0: 
      save_metadata(protect=isFirstSave, sort=False)
      isFirstSave = False
  
  # Save metadata (if necesary)
  if created > 0: save_metadata(protect=isFirstSave)

def fix_all_metadata():
  # Temp vars
  isFirstSave = True
  fixes = 0

  # Loop images with metadata
  size = len(images_with_metadata)
  for i in reversed(range(0, size)):
    # Get image path & load its metadata
    image_path = join(images_folder, images_with_metadata[i])
    image_metadata = metadata[images_with_metadata[i]]

    # No need for a fix
    if not metadata_needs_fix(image_metadata): continue

    # Load image
    image_number = size - i
    print('\nFixing (' + str(size - i) + ' of ' + str(size) + '): ' + image_path)
    image = Image.open(image_path)

    # Fix metadata
    fix_image_metadata(image, image_metadata)
    fixes += 1

    # Save metadata to prevent losing progress (without sorting, is faster)
    if image_number != size and fixes % save_every == 0: 
      save_metadata(protect=isFirstSave, sort=False)
      isFirstSave = False

  # Log results
  print('Fixed ' + str(fixes) + ' metadata keys')
  
  # Save metadata (if necesary)
  if fixes > 0: save_metadata(protect=isFirstSave)

# Metadata file
def sort_and_clean_metadata():
  global metadata

  # Create new metadata
  new_metadata = {}

  # Sort images by modified date
  images_with_metadata.sort(key=lambda image_name: getmtime(join(images_folder, image_name)), reverse=True)

  # Add image keys that exist
  for image_name in images_with_metadata:
    new_metadata[image_name] = metadata[image_name]

  # Replace old metadata with the new one
  metadata = new_metadata

def protect_metadata(base_path, number=0):
  # Get real path
  metadata_path = (base_path + '.' + str(number)) if (number != 0) else (base_path)
  metadata_path_next = (base_path + '.' + str(number + 1))

  # Protect next file
  if (isfile(metadata_path)):
    protect_metadata(base_path, number + 1)
    rename(metadata_path, metadata_path_next)

def save_metadata(protect=True, sort=True):
  # Print action
  if sort:
    print('Saving metadata...')
  else:
    print('Saving metadata (unsorted)...')

  # Sort & clean before saving
  if sort: sort_and_clean_metadata()

  # Protect metadata file (create a backup)
  if protect: protect_metadata(metadata_file)

  # Save metadata
  Util.save_json(metadata_file, metadata)




#   /$$$$$$
#  /$$__  $$
# | $$  \ $$  /$$$$$$   /$$$$$$   /$$$$$$$
# | $$$$$$$$ /$$__  $$ /$$__  $$ /$$_____/
# | $$__  $$| $$  \__/| $$  \ $$|  $$$$$$ 
# | $$  | $$| $$      | $$  | $$ \____  $$
# | $$  | $$| $$      |  $$$$$$$ /$$$$$$$/
# |__/  |__/|__/       \____  $$|_______/ 
#                      /$$  \ $$
#                     |  $$$$$$/
#                      \______/

for arg in sys.argv:
  # Not a valid argument
  if not arg.startswith('--'): continue

  # Get argument name & content
  arg_name = arg[2:]
  arg_content = ''
  if ':' in arg_name:
    index = arg_name.index(':')
    arg_content = arg_name[index+1:]
    arg_name = arg_name[:index]

  # Check name
  match arg_name:
    # Images folder
    case 'images':
      if arg_content.startswith('"') and arg_content.endswith('"'):
        images_folder = arg_content[1:-1]
      else:
        images_folder = arg_content
    # Metadata file
    case 'metadata':
      if arg_content.startswith('"') and arg_content.endswith('"'):
        metadata_file = arg_content[1:-1]
      else:
        metadata_file = arg_content



#  /$$      /$$                              
# | $$$    /$$$                              
# | $$$$  /$$$$  /$$$$$$  /$$$$$$$  /$$   /$$
# | $$ $$/$$ $$ /$$__  $$| $$__  $$| $$  | $$
# | $$  $$$| $$| $$$$$$$$| $$  \ $$| $$  | $$
# | $$\  $ | $$| $$_____/| $$  | $$| $$  | $$
# | $$ \/  | $$|  $$$$$$$| $$  | $$|  $$$$$$/
# |__/     |__/ \_______/|__/  |__/ \______/ 

# Ask for images path
if images_folder == '':
  # Get path from console input
  images_folder = input('Images path: ')
  if not isdir(images_folder):
    exit()

# Ask for metadata path
if metadata_file == '':
  # Get path from console input
  metadata_file = input('Metadata path: ')
  if not isfile(metadata_file):
    exit()

# Load metadata file
metadata = Util.load_json(metadata_file)

# Create lists (with and without metadata) of images in folder
print('\nLoading images lists...')
Util.load_images(images_folder, metadata, images_with_metadata, images_without_metadata)

# Declare show menu function
def show_menu():
  print('\nMENU')
  print('0. Exit')
  print('1. Search')
  print('2. Create missing metadata')
  print('3. Fix incomplete metadata')
  print('4. Sort & clean metadata')

# Show menu option picker
option = -1
while option < 0 or option > 4:
  show_menu()
  option = int(input('Choose an option: '))

# Run option
if option == 1:
  search()
elif option == 2:
  create_all_metadata()
elif option == 3:
  fix_all_metadata()
elif option == 4:
  save_metadata_sorted_and_clean()