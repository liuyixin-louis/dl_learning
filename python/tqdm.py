import time
from tqdm import tqdm
from tqdm._tqdm import trange
 
for i in tqdm(range(100)):
    time.sleep(0.01)