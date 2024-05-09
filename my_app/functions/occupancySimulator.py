import pandas as pd
import numpy as np

class OccupancySimulator:

    def __init__(self, employees=400, base_percentages=None, noise_scales=None, sign=-1):

        self.employees = employees
        self.base_percentages = base_percentages or {
            'Monday': 30,
            'Tuesday': 20,
            'Wednesday': 10,
            'Thursday': 10,
            'Saturday': 80,
            'Sunday': 90,
            'Christmas': 55,
            'August': 60
        }
        self.noise_scales = noise_scales or {
            'Weekdays': 10,
            'Weekends': 5,
            'Christmas': 10,
            'August': 10
        }
        self.sign = sign or -1


    def simulate_occupancy(self, date):

        day_of_week = date.day_name()

        if (date.month == 12 and date.day >= 22) or (date.month == 1 and date.day <= 6):
            noise = np.random.normal( loc=0, scale=self.noise_scales['Christmas'] )
            base_percentage = self.base_percentages['Christmas']
        elif (date.month == 8 and date.day >= 10) or (date.month == 8 and date.day <= 24):
            noise = np.random.normal( loc=0, scale=self.noise_scales['August'] )
            base_percentage = self.base_percentages['August']
        else:
            noise = np.random.normal(loc=0, scale=self.noise_scale[day_of_week])
            base_percentage = self.base_percentages[day_of_week]
        
        if self.sign >= 0:
            percentage_with_noise = max(min(self.sign*base_percentage + noise, 100), 0)
        else:
            percentage_with_noise = min(max(self.sign*base_percentage + noise, -100), 0)
        
        occupants = self.employees  + round(percentage_with_noise / 100 * self.employees)

        return occupants
    

    def simulate_occupancy_for_data(self, data):

        for index, values in data.iterrows():

            date = pd.to_datetime(index)

            occupants = self.simulate_occupancy(self, date)

            data.loc[index, 'occupants'] = occupants
        
        return data



