# URL of the HTML page containing the table
root_url = "https://www.wormatlas.org/neurons/Individual%20Neurons/"
all_neurons_url = "https://www.wormatlas.org/neurons/Individual%20Neurons/Neuronframeset.html"

import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
import re

# List of all 302 hermaphrodite neurons
NEURONS_302 = sorted(
    pd.read_csv(
        "/home/lrvnc/Projects/worm-graph/data/raw/neurons_302.txt",
        sep=" ",
        header=None,
        names=["neuron"],
    ).neuron
)

neuron_classification = {}

# Set up the web driver
driver = webdriver.Chrome()  # Replace with the appropriate web driver for your browser
wait = WebDriverWait(driver, 10)  # Wait up to 10 seconds

# Navigate to the main webpage
driver.get(all_neurons_url)

for neuron in NEURONS_302:
    neuron_group = neuron[:3]

    if len(neuron_group) == 2:
         continue
    elif (neuron_group.startswith('P') or neuron_group.startswith('A')) and (neuron_group.endswith('R') or neuron_group.endswith('L')):
         continue
    elif neuron_group == 'RIR':
        continue
    elif neuron_group[-1] == 'R' or neuron_group[-1] == 'L' and (neuron_group.startswith('I') or neuron_group.startswith('M')):
        neuron_group = neuron_group[:2]
    elif neuron_group[-1].isdigit() and not neuron_group.startswith('I'):
            neuron_group = neuron_group[:-1] + 'N'

    neuron_url = 'https://www.wormatlas.org/neurons/Individual%20Neurons/' + neuron_group + 'mainframe.htm'

    # Open a new window and switch to it
    driver.execute_script("window.open('about:blank', 'new_window')")
    driver.switch_to.window(driver.window_handles[-1])

    # Navigate to the new website
    driver.get(neuron_url)

    # Scrape the text from the new page
    text_element = wait.until(EC.presence_of_element_located((By.XPATH, "/html/body/table[2]/tbody/tr/td/table[1]/tbody/tr/td/table[1]/tbody/tr/td[1]")))
    text = text_element.text
    time.sleep(0.5)

    # Close the new window
    driver.close()

    # Switch back to the first page
    driver.switch_to.window(driver.window_handles[0])

    # Extract Type
    match = re.search(r"Type:\s+(.*)", text)
    type_content = match.group(1)
    print("Neuron: {}, Type: {}".format(neuron, type_content))

    neuron_classification[neuron] = type_content
# Perform additional actions or extract information from the subpages as needed

# Close the web driver
driver.quit()

import json
json_string = json.dumps(neuron_classification)
with open("neuron_classification_raw.json", "w") as file:
    file.write(json_string)


new_data = {}

for neuron, description in neuron_classification.items():

    new_classification = []
    # Just PDB -> Motor neuron in the hermaphrodite, motor neuron and interneuron in the male

    if 'sensory' in description.lower():
        new_classification.append('sensory')
    
    if 'motor' in description.lower():
        new_classification.append('motor')

    if 'interneuron' in description.lower():
        new_classification.append('interneuron')

    if 'unknown' in description.lower():
        new_classification.append('unknown')

    new_data[neuron] = new_classification

json_string = json.dumps(new_data)

with open("neuron_clusters.json", "w") as file:
    file.write(json_string)