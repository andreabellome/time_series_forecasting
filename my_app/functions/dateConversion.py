import numpy as np

class DateConversion:

    def __init__(self):
        pass

    def hms2fracday(self, hrs: float, mn: float, sec: float):
        return (hrs + (mn + sec/60)/60) / 24

    def date2jd(self, date_str: str, format: str = '%Y-%m-%d %H:%M:%S'):
        # from datetime64 to string
        Y = float(date_str[0:4])
        M = float(date_str[5:7])
        D = float(date_str[8:10])
        hrs = float(date_str[11:13])
        mn = float(date_str[14:16])
        sec = float(date_str[17:19])

        jd = 367.0*Y - np.floor(7*(Y+np.floor((M+9)/12))/4) - np.floor(3*np.floor((Y+(M-9)/7)/100+1)/4) + np.floor(275*M/9) + D + 1721028.5 + self.hms2fracday(hrs, mn, sec)

        return jd
