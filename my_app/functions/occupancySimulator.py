import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class OccupancySimulator:

    def __init__(self, employees=400, base_percentages=None, noise_scales=None, sign=-1):

        self.employees = employees
        self.base_percentages = base_percentages or {
            'Monday': 45,
            'Tuesday': 40,
            'Wednesday': 35,
            'Thursday': 35,
            'Friday': 35,
            'Saturday': 70,
            'Sunday': 80,
            'Christmas': 85,
            'August': 70
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

        if day_of_week in ['Saturday', 'Sunday']:
            scale = self.noise_scales['Weekends']
        else:
            scale = self.noise_scales['Weekdays']

        noise = np.random.normal(loc=0, scale=scale)
        base_percentage = self.base_percentages[day_of_week]

        if self.sign >= 0:
            percentage_with_noise = max(min(self.sign*base_percentage + noise, 100), 0)
        else:
            percentage_with_noise = min(max(self.sign*base_percentage + noise, -100), 0)

        occupants = self.employees + round(percentage_with_noise / 100 * self.employees)

        if (date.month == 12 and date.day >= 22) or (date.month == 1 and date.day <= 6):

            noise = np.random.normal( loc=0, scale=self.noise_scales['Christmas'] )
            base_percentage = self.base_percentages['Christmas']
            if self.sign >= 0:
                percentage_with_noise = max(min(self.sign*base_percentage + noise, 100), 0)
            else:
                percentage_with_noise = min(max(self.sign*base_percentage + noise, -100), 0)

            occupants = self.employees + round(percentage_with_noise / 100 * self.employees)

        elif (date.month == 8 and date.day >= 10) or (date.month == 8 and date.day <= 24):
            noise = np.random.normal( loc=0, scale=self.noise_scales['August'] )
            base_percentage = self.base_percentages['August']
            if self.sign >= 0:
                percentage_with_noise = max(min(self.sign*base_percentage + noise, 100), 0)
            else:
                percentage_with_noise = min(max(self.sign*base_percentage + noise, -100), 0)

            occupants = self.employees + round(percentage_with_noise / 100 * self.employees)
        
        return occupants
    

    def generate_datetime_range(self, start_date, end_date, step):
        
        current_date = start_date
        datetime_list = []
        while current_date <= end_date:
            datetime_list.append(current_date)
            current_date += timedelta(days=step)
        return datetime_list

    def simulate_occupancy_future( self, start_date, end_date, step, columnName:str = 'occupants' ):
        
        datetime_list = self.generate_datetime_range( start_date, end_date, step )

        occupants_list = []
        for date in datetime_list:

            occupants = self.simulate_occupancy( date )
            occupants_list.append(occupants)

        data = pd.DataFrame({'Date': datetime_list, columnName: occupants_list})

        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        data = data.set_index('Date')
        data = data.asfreq('1D')
        data = data.sort_index()

        return data

    def simulate_occupancy_for_data(self, data):

        for index, values in data.iterrows():

            date = pd.to_datetime(index)

            occupants = self.simulate_occupancy(date)

            data.loc[index, 'occupants'] = occupants
        
        return data



