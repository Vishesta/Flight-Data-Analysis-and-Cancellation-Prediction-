import os
import settings
import pandas as pd



def concatenate():

    # load csv file containing the airport information
    airport_df = pd.read_csv(os.path.join(settings.DATA_DIR,"airports.csv"))

    #load csv file containing the flight delay information. this will be the main dataframe
    flights_df =  pd.read_csv(os.path.join(settings.DATA_DIR,"flights.csv"))
    main_df = flights_df
    # combine these, so that the main dataframe has latitude longitude information
    airport_df = airport_df.set_index("IATA_CODE")
    for loc in ['ORIGIN','DESTINATION']:
        for dir in ['LATITUDE','LONGITUDE']:
            main_df[loc+'_'+dir] =(airport_df[dir][flights_df[loc+"_AIRPORT"][:]]).reset_index()[[dir]]
    main_df.to_csv(os.path.join(settings.PROCESSED_DIR, "{}.txt".format('processed')), index=False)


if __name__ == "__main__":
    concatenate()
