# -*- coding: utf-8 -*-
"""

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CYfW2Tj35SZISCY8KigNB2YtwUgN1SJN
"""

import json
import requests
from geopy.geocoders import Nominatim
from city_name_coordinates_utils import *

def main():
    
    city_coordinates_data = get_city_coordinates(city_list, save_to_json=True)
    
    print(city_coordinates_data)

    questions_city_example, questions_num_example, answers_example = generate_equations("city_coordinates.json", 30, 4, flag=1)
    
    print(questions_city_example)
    print(questions_num_example)
    print(answers_example)


 
    verification_result_example = verify_equations("arithmetic_equations_longitude.json", "city_coordinates.json", flag=1)

    print(verification_result_example)

    questions_city_example, questions_num_example, answers_example = generate_equations("city_coordinates.json", 30, 4, flag=2)
    
    print(questions_city_example)
    print(questions_num_example)
    print(answers_example)

    verification_result_example = verify_equations("arithmetic_equations_latitude.json", "city_coordinates.json", flag=2)

    print(verification_result_example)

if __name__ == "__main__":
    # Define a list of 20 big cities from different continents
    city_list = [
        "New York", "Los Angeles", "London", "Tokyo", "Beijing",
        "Sydney", "Cairo", "Sao Paulo", "Mumbai", "Moscow",
        "Lagos", "Johannesburg", "Buenos Aires", "Paris", "Istanbul",
        "Seoul", "Bangkok", "Rome", "Toronto", "Mexico City"
    ]

    main()