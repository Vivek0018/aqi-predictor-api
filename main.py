from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from datetime import datetime, timedelta
from sseclient import SSEClient
import json
import js2py
# import seaborn as sns
from io import BytesIO
import base64


from sktime.forecasting.base import ForecastingHorizon
# from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,mean_squared_percentage_error,median_absolute_percentage_error
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.fbprophet import Prophet
# from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
import matplotlib.pyplot as plt
import numpy as np


class AQIData:
    def __init__(self):
        self.JS_FUNCS: str = """
        function checkValidDigitNumber(t) {
            return !isNaN(t) && parseInt(Number(t)) == t && !isNaN(parseInt(t, 10))
        }

        function a(backendData, n) {
            var e = 0,
                i = 0,
                r = 0,
                o = 1,
                resultArray = [];

            function s(t, r) {
                /* Variable r seems to uselessly bounce from 0 to 1 to 0 for no reason
                other than to obfuscate

                If r is 0 the code executes, otherwise it won't */

                for (0 == r && (r = 1); r > 0; r--) e++, i += t, resultArray.push({
                    t: n(e), /** n seems to be a method to determine "which day of month" */
                    v: i * o /** appears to be "value"? */
                })
            }

            function charInPositionIsDigit(t) {
                /* ASCII 48-57 is for 0-9 (digits) */
                return backendData.charCodeAt(t) >= 48 && backendData.charCodeAt(t) <= 57
            }
            for (var idx = 0; idx < backendData.length; idx++) {
                var u = function() {
                        var t = 0,
                            n = 1;
                            /** 45 is ASCII for - and 46 is ASCII for . */
                        for (45 == backendData.charCodeAt(idx + 1) && (n = -1, idx++); charInPositionIsDigit(idx + 1);) t = 10 * t + (backendData.charCodeAt(idx + 1) - 48), idx++;
                        return 46 == backendData.charCodeAt(idx + 1) && idx++, n * t
                    },
                    h = backendData.charCodeAt(idx);
                if (0 == idx && 42 == h) o = 1 / u(), idx++;    /* 42 is ASCII for * */
                else if (36 == h) e += 1;           /* 36 is ASCII for $ */
                else if (37 == h) e += 2;           /* 37 is ASCII for % */
                else if (39 == h) e += 3;           /* 39 is ASCII for ' */
                else if (47 == h) o = u(), idx++;     /* 47 is ASCII for / */
                else if (33 == h) s(u(), r), r = 0; /* 33 is ASCII for ! */
                else if (124 == h) e += u() - 1;    /* 124 is ASCII for | */
                else if (h >= 65 && h <= 90) s(h - 65, r), r = 0;           /* This conditional is true when given ASCII for uppercase A-Z */
                else if (h >= 97 && h <= 122) s(-(h - 97) - 1, r), r = 0;   /* This conditional is true when given ASCII for lowercase a-z */
                else {
                    if (!(h >= 48 && h <= 57)) throw "decode: invalid character " + h + " (" + backendData.charAt(idx) + ") at " + idx;
                    r = 10 * r + h - 48
                }
            }
            return resultArray
        }

        function s(t) {
            /* NOTE: Appears to be the "main gun" since here's a try catch block */
            if (!t) return null;
            try {
                var n, e, i = [],
                    r = {
                        pm25: "PM<sub>2.5</sub>",
                        pm10: "PM<sub>10</sub>",
                        o3: "O<sub>3</sub>",
                        no2: "NO<sub>2</sub>",
                        so2: "SO<sub>2</sub>",
                        co: "CO"
                    },
                    o = function() {
                        try {
                            n = [];
                            var o = t.ps[s]; /* Long string backend data is o */
                            if ("1" == o[0]) n = a(o.substr(1), function(n) {
                                return {
                                    d: c(new Date(3600 * (n * t.dh + t.st) * 1e3)), /** This expression results in 'seconds after Unix epoch' style value. st is an "hour after Unix epoch" value. */
                                    t: n
                                }
                            });
                            else if ("2" == o[0]) {
                                e = {};
                                var d = "w" == o[1];
                                for (var l in o.substr(3).split("/").forEach(function(n) {
                                        a(n, function(n) {
                                            if (d) {
                                                var e = n + t.st,
                                                    i = e % 53;
                                                return {
                                                    d: c(function(t, n, e) {
                                                        var i = 2 + e + 7 * (n - 1) - new Date(t, 0, 1).getDay();
                                                        return new Date(t, 0, i)
                                                    }(a = (e - i) / 53, i, 0)),
                                                    t: n
                                                }
                                            }
                                            var r = n + t.st,
                                                o = r % 12,
                                                a = (r - o) / 12;
                                            return {
                                                d: c(new Date(a, o)),
                                                t: n
                                            }
                                        }).forEach(function(t) {
                                            var n = t.t.t;
                                            e[n] = e[n] || {
                                                v: [],
                                                t: t.t
                                            }, e[n].v.push(t.v)
                                        })
                                    }), e) n.push(e[l])
                            }
                            n.forEach(function(t, e) {
                                n[e].t.dh = e ? (t.t.d.getTime() - n[e - 1].t.d.getTime()) / 36e5 : 0
                            }), i.push({
                                name: r[s] || s,
                                values: n,
                                pol: s
                            })
                        } catch (t) {
                            console.error("decode: Oopps...", t)
                        }
                    };
                for (var s in t.ps) o(); /* For each variable? do o()*/
                return i.sort(function(t, n) {
                    var e = ["pm25", "pm10", "o3", "no2", "so2", "co"],
                        i = e.indexOf(t.pol),
                        r = e.indexOf(n.pol);
                    return r < 0 ? 1 : i < 0 ? -1 : i - r
                }), {
                    species: i,
                    dailyhours: t.dh,
                    source: t.meta.si,
                    period: t.period
                }
            } catch (t) {
                return console.error("decode:", t), null
            }
        }

        function c(t) {
            return new Date(t.getUTCFullYear(), t.getUTCMonth(), t.getUTCDate(), t.getUTCHours(), t.getUTCMinutes(), t.getUTCSeconds())
        }

        function gatekeep_convert_date_object_to_unix_seconds(t) {
            /** Wrapper function:
                Perform decoding using s() function above, and afterwards convert all Date objects within
                the result into Unix timestamps, i.e. 'seconds since 1970/1/1'.
                This is necessary so that the Python context can convert that Unix timestamps back into datetime objects.
                js2py is unable to (at the time of writing, to my limited knowledge) convert JS Date objects into Python-understandable objects.
            */
            var RES = s(t)
            for(var i = 0; i < RES.species.length; i++){
            var values = RES.species[i].values
                for(var j = 0; j < values.length; j++){
                    values[j].t.d = values[j].t.d.getTime()/1000
                }
            RES.species[i].values = values
            }
            return RES
        }
        """


        # NOTE(lahdjirayhan):
        # The JS_FUNCS variable is a long string, a source JS code that
        # is excerpted from one of aqicn.org frontend's scripts.
        # See relevant_funcs.py for more information.


        # Make js context where js code can be executed
        self._context = js2py.EvalJs()
        self._context.execute(self.JS_FUNCS)

    # def __init__(self, token:str = '') -> None:
    #     self.token = token

    def parse_incoming_result(self, json_object: dict) -> pd.DataFrame:

        print('Parsing the results from API')
        # Run JS code
        # Function is defined within JS code above
        # Convert result to Python dict afterwards
        OUTPUT = self._context.gatekeep_convert_date_object_to_unix_seconds(
            json_object["msg"]
        ).to_dict()

        result_dict = {}
        for spec in OUTPUT["species"]:
            pollutant_name: str = spec["pol"]

            dates, values = [], []
            for step in spec["values"]:
                # Change unix timestamp back to datetime
                date = datetime.fromtimestamp(step["t"]["d"])
                value: int = step["v"]

                dates.append(date)
                values.append(value)

            series = pd.Series(values, index=dates)
            result_dict[pollutant_name] = series

        FRAME = pd.DataFrame(result_dict)
        return FRAME

        
    def get_results_from_backend(self, city_id: int):
        print("Geting results from API")
        event_data_url = f"https://api.waqi.info/api/attsse/{city_id}/yd.json"

        r = requests.get(event_data_url)

        # Catch cases where the returned response is not a server-sent events,
        # i.e. an error.
        if "text/event-stream" not in r.headers["Content-Type"]:
            raise Exception(
                "Server does not return data stream. "
                f'It is likely that city ID "{city_id}" does not exist.'
            )

        client = SSEClient(event_data_url)
        result = []

        for event in client:
            if event.event == "done":
                break

            try:
                if "msg" in event.data:
                    result.append(json.loads(event.data))
            except json.JSONDecodeError:
                pass

        return result


    def get_data_from_id(self, city_id: int) -> pd.DataFrame:
        backend_data = self.get_results_from_backend(city_id)
        result = pd.concat([self.parse_incoming_result(data) for data in backend_data])
        # result = parse_incoming_result(backend_data[0])

        # Arrange to make most recent appear on top of DataFrame
        result = result.sort_index(ascending=False, na_position="last")

        # Deduplicate because sometimes the backend sends duplicates
        result = result[~result.index.duplicated()]

        # Reindex to make missing dates appear with value nan
        # Conditional is necessary to avoid error when trying to
        # reindex empty dataframe i.e. just in case the returned
        # response AQI data was empty.
        if len(result) > 1:
            complete_days = pd.date_range(
                result.index.min(), result.index.max(), freq="D"
            )
            result = result.reindex(complete_days, fill_value=None)

            # Arrange to make most recent appear on top of DataFrame
            result = result.sort_index(ascending=False, na_position="last")

        return result

    
    def get_city_station_options(self, city: str) -> pd.DataFrame:
        print('Getting Station options in the city')
        """Get available stations for a given city
        Args:
            city (str): Name of a city.

        Returns:
            pd.DataFrame: Table of stations and their relevant information.
        """
        # NOTE, HACK, FIXME:
        # This functionality was born together with historical data feature.
        # This endpoint is outside WAQI API's specification, thus not using
        # _check_and_get_data_obj private method above.
        # If exists, alternative within API's spec is more than welcome to
        # replace this implementation.
        r = requests.get(f"https://search.waqi.info/nsearch/station/{city}")
        res = r.json()

        city_id, country_code, station_name, city_url, score = [], [], [], [], []

        for candidate in res["results"]:
            city_id.append(candidate["x"])
            country_code.append(candidate["c"])
            station_name.append(candidate["n"])
            city_url.append(candidate["s"].get("u"))
            score.append(candidate["score"])

        return pd.DataFrame(
            {
                "city_id": city_id,
                "country_code": country_code,
                "station_name": station_name,
                "city_url": city_url,
                "score": score,
            }
        ).sort_values(by=["score"], ascending=False)


    def get_historical_data(
        self, city: str = None, city_id: int = None  # type: ignore
    ) -> pd.DataFrame:
        
        print(f'Getting Historical Data of {city}')
        """Get historical air quality data for a city

        Args:
            city (str): Name of the city. If given, the argument must be named.
            city_id (int): City ID. If given, the argument must be named.
                If not given, city argument must not be None.

        Returns:
            pd.DataFrame: The dataframe containing the data.
        """
        if city_id is None:
            if city is None:
                raise ValueError("If city_id is not specified, city must be specified.")

            # Take first search result
            search_result = self.get_city_station_options(city)
            if len(search_result) == 0:
                return 404

            first_result = search_result.iloc[0, :]

            city_id = first_result["city_id"]
            station_name = first_result["station_name"]
            country_code = first_result["country_code"]

        df = self.get_data_from_id(city_id)
        if "pm25" in df.columns:
            # This ensures that pm25 data is labelled correctly.
            df.rename(columns={"pm25": "pm2.5"}, inplace=True)

        # Reset date index and rename the column appropriately
        # df = df.reset_index().rename(columns={"index": "date"})
        # print(df)

        return [df ,city , station_name, country_code]
    

# class Forecasters(dataset):
#     def __init__(self):
#         self.dataset = dataset
    
#     def Prophet(self):
#         forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)


def sktime_forecast(dataset, horizon=30, validation=False, confidence=0.9, frequency="D"):
    """Loop over a time series dataframe, train an sktime forecasting model, and visualize the results.

    Args:
        dataset (pd.DataFrame): Input time series DataFrame with datetime index
        horizon (int): Forecast horizon
        forecaster (sktime.forecasting): Configured forecaster
        validation (bool, optional): . Defaults to False.
        confidence (float, optional): Confidence level. Defaults to 0.9.
        frequency (str, optional): . Defaults to "D".
    """

    forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    # Adjust frequency of index(dates)
    forecast_df = dataset.resample(rule=frequency).sum()
    # Interpolate missing periods (if any)
    forecast_df = forecast_df.interpolate(method="time")

    all_parameters_values = {}

    #to store plot images
    plotImages = {}
    for col in dataset.columns:
        # Use train/test split to validate forecaster
        if validation:
            df = forecast_df[col]

            y_train = df[:-horizon]
            y_test = df.tail(horizon)

            forecaster.fit(y_train)
            fh = ForecastingHorizon(y_test.index, is_relative=False)
            y_pred = forecaster.predict(fh)
            ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")
            y_true = df.tail(horizon)

            # mae = mean_absolute_error(y_true, y_pred)

        # Make predictions beyond the dataset
        if not validation:
            df = forecast_df[col].dropna()
          
            forecaster.fit(df)

            #for present date            
            present_date = datetime.now().date()
            #to start predictions from tomorrow
            present_date = str(present_date + timedelta(days=1)).split(' ')[0]
            fh = ForecastingHorizon(
                pd.date_range(str(present_date), periods=horizon, freq=frequency),
                is_relative=False,
            )

            y_pred = forecaster.predict(fh)
            ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")
            # mae = np.nan

        # Visualize results
        # plt.plot(
        #     df.tail(horizon),
        #     label="Actual",h
        #     color="black",
        # )
        # plt.gca().fill_between(
        #     ci.index, (ci.iloc[:, 0]), (ci.iloc[:, 1]), color="b", alpha=0.1
        # )
        # print(y_pred)
        # plt.imshow(y_pred, cmap='hot', interpolation='nearest')
        # # plt.plot(y_pred, label="Predicted")
        # # plt.xticks(rotation=30, ha='right')
        # # # plt.title(
        # # #     f"{horizon} day forecast for {col} (mae: {round(mae, 2)}, confidence: {confidence*100}%)"
        # # # )
        # # plt.ylim(bottom=0)
        # # plt.legend()
        # # plt.grid(False)
        # plt.show()
        # print(y_pred)
            
        # # data = np.random.rand(10, 10)

        # # Create heatmap
        # # plt.imshow(data, cmap='hot', interpolation='nearest')
        # # plt.colorbar()  # Add color bar indicating the scale
        # # plt.show()

        # plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
        # sns.heatmap(y_pred, annot=True, cmap='coolwarm', linewidths=.5)
        # plt.title('Heatmap of DataFrame')
        # plt.show()

        # buffer = BytesIO()
        # # plt.savefig(buffer, format='png')
        # buffer.seek(0)
        # image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        # buffer.close()

        # plotImages[col] = image_base64
        # print(image_base64)
        # print("Mean Absolute Error : ", mae)

        # try :
        #     temp = all_parameters_values['date']
        # except:
        #     all_parameters_values['date'] = [i.strftime("%d-%m-%Y") for i in fh]

        all_parameters_values[col] = y_pred.values
    

    
    dates = [i.strftime("%d-%m-%Y") for i in fh]

    predicted_data = {}
    for date in range(len(dates)):
        temp = {}
        for param in all_parameters_values:
            temp[param] = all_parameters_values[param][date]
        predicted_data[dates[date]] = temp
    
    return [predicted_data, plotImages]



class Sktime_forecast:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.frequency = 'D'
        self.dataset = self.dataset.resample(rule=self.frequency).sum()
        print(self.dataset)
        # Interpolate missing periods (if any)
        self.dataset = self.dataset.interpolate(method="time")
        print(self.dataset)
        self.horizon = 30
        # self.validation = False
        self.confidence = 0.9 

        self.y_train = self.dataset[:-30]
        self.y_test = self.dataset.tail(30)

        self.fh_train = ForecastingHorizon(self.y_test.index, is_relative=False)

        #for present date
        present_date = datetime.now().date()
        #to start predictions from tomorrow
        present_date = str(present_date + timedelta(days=1)).split(' ')[0]

        self.fh_pred = ForecastingHorizon(pd.date_range(str(present_date), periods=30, freq='D'),is_relative=False)

    def getAccuracyMetrics(self, forecaster, forecaster_name):
        print('-----------------------------------------------------------')
        print(f'\nStarted Calculating Accuracy metrics for {forecaster_name}')
        performance_metrics = {}
        for param in self.y_train:
            forecaster.fit(self.y_train[param])
            y_pred = forecaster.predict(self.fh_train)
            # ci = forecaster.predict_interval(fh, coverage=0.9).astype("float")
            y_true = self.dataset[param].tail(30)

            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mspe = mean_squared_percentage_error(y_true, y_pred)    
            mape = median_absolute_percentage_error(y_true, y_pred)
            # r2_score_ = r2_score(y_true, y_pred)

            performance_metrics[param] = {
                'mae': mae,
                'mape': mape,
                'mse': mse,
                'rmse': rmse,
                'mspe': mspe,
                'mape': mape
            }

        print(f'Calculated Accuracy metrics for {forecaster_name}')
        return performance_metrics
    
    def getPredictions(self, forecaster, forecaster_name):
        print(f'Started making Predictions using {forecaster_name} Model')
        predictions = {}
        plots = {}

        for param in self.dataset:
            forecaster.fit(self.dataset[param])

            y_pred = forecaster.predict(self.fh_pred)

            # predictions[param] = y_pred

            for i in self.fh_pred:
                try:
                    predictions[i.strftime('%Y-%m-%d')][param] = y_pred[i]
                except:
                    predictions[i.strftime('%Y-%m-%d')] = {}
                    predictions[i.strftime('%Y-%m-%d')][param] = y_pred[i]
            

            # to clear the plot
            plt.clf()

            plt.figure()
            # Visualize results
            plt.plot(
                self.dataset[param].tail(100),
                label="Actual",
                color='black'
            )
            # plt.gca().fill_between(
            #     ci.index, (ci.iloc[:, 0]), (ci.iloc[:, 1]), color="b", alpha=0.1
            # )
            # plt.imshow(y_pred, cmap='hot', interpolation='nearest')
            plt.plot(
                y_pred, 
                label="Predicted",
            )
            plt.xticks(rotation=30, ha='right')
            # plt.title(
            #     f"{horizon} day forecast for {col} (mae: {round(mae, 2)}, confidence: {confidence*100}%)"
            # )
            plt.ylim(bottom=0)
            plt.legend()
            plt.grid(False)

            # plt.show()

            #converting plot into a image
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()

            plots[param] = image_base64


        print(f'Predictions using {forecaster_name} Model are Done!')
        print('-----------------------------------------------------------')

        return predictions, plots
    

    def getAQIForecasts(self, model):
        print('Forecaster Started')
        
        forecasters = {
            'prophet' :{
                'name' : 'Prophet',
                'forecaster' : Prophet(yearly_seasonality=True, weekly_seasonality=True),
            },
            'exponentialsmoothening' : {
                'name' : 'Exponential Smoothening',
                'forecaster': ExponentialSmoothing(trend="mul", seasonal="mul", sp=12)
            },
            'autoarima': {
                'name' : 'ARIMA (Auto Regressive Integrated Moving Average)',
                'forecaster' : AutoARIMA(sp=1, suppress_warnings=True)
            }
        }


        predictions = self.getPredictions(forecasters[model]['forecaster'])
        forecasts = {
            'code': 200,
            'model_name' : forecasters[str.lower(model)]['name'],
            'accuracy_metrics' : self.getAccuracyMetrics(forecasters[model]['forecaster']),
            'predictions' : predictions[0],
        }

        # for forecaster_name, forecaster in forecasters.items():
        #     forecasts[forecaster_name] = {}
        #     forecasts[forecaster_name] = {}
        #     forecasts[forecaster_name]['accuracy_metrics'] = self.getAccuracyMetrics(forecaster)
        #     forecasts[forecaster_name]['predictions'] = self.getPredictions(forecaster)
        
        print('Forecasters Completed !')
        return forecasts
    
    def getAllAQIForecastsAtATime(self):
        
        forecasters = {
            'Prophet' : Prophet(yearly_seasonality=True, weekly_seasonality=True),
            'ExponentialSmoothening' : ExponentialSmoothing(trend="add", seasonal="add", sp=12),
            'AutoARIMA': AutoARIMA(sp=1, suppress_warnings=True)
        }

        forecasts = {}


        for forecaster_name, forecaster in forecasters.items():
            forecasts[forecaster_name] = {}
            forecasts[forecaster_name] = {}
            forecasts[forecaster_name]['accuracy_metrics'] = self.getAccuracyMetrics(forecaster, forecaster_name)
            predictions = self.getPredictions(forecaster, forecaster_name)

            forecasts[forecaster_name]['predictions'] = predictions[0]
            forecasts[forecaster_name]['plots'] = predictions[1]
        
        return forecasts

def getOnlyCityData(city_name):
    #creating AQI data object
    o = AQIData()
    # dataset = o.get_historical_data(city="New York")
    # forecaster = AutoARIMA(sp=1, suppress_warnings=True)

    #creating forecaster Object
    # forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    # forecaster = ThetaForecaster(sp=12)

    #getting historical data of the city
    data = o.get_historical_data(city=city_name)
    
    #for storing final output
    finalOut = {}

    #if data exists about the city
    if data != 404:

        dates = [i for i in data[0].index]
        for i in range(len(dates)):
            if(dates[i].date() > datetime.now().date()):
                data[0].drop(index=dates[i].date(), inplace=True)

        dataset = data[0]
        dataset = dataset.dropna()

        return dataset


def getProphetData(city_name):



    return

def getCityData(city_name):
    #creating AQI data object
    o = AQIData()
    # dataset = o.get_historical_data(city="New York")
    # forecaster = AutoARIMA(sp=1, suppress_warnings=True)

    #creating forecaster Object
    # forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    # forecaster = ThetaForecaster(sp=12)

    #getting historical data of the city
    data = o.get_historical_data(city=city_name)
    
    #for storing final output
    finalOut = {}

    #if data exists about the city
    if data != 404:

        dates = [i for i in data[0].index]
        for i in range(len(dates)):
            if(dates[i].date() > datetime.now().date()):
                data[0].drop(index=dates[i].date(), inplace=True)

        dataset = data[0]
        dataset = dataset.dropna()

        print(dataset)

        dataset.to_csv('Data/dataset.csv', index=False)
        # return dataset
    
        models = Sktime_forecast(dataset)

        predictions = models.getAllAQIForecastsAtATime()

        #saving the file locally without index
        # dataset.to_csv(f"Data/{city_name}_data.csv", index=False)

        #reading the file while parsing the dates
        # dataset = pd.read_csv(f"Data/{city_name}_data.csv", parse_dates=[0], index_col=[0])
        # print(dataset)
        #remove future dates in the dateset
        

        # t = sktime_forecast(dataset=dataset, horizon=30, validation=False)

        # predicted_data, plotImages = t[0], t[1]

        #for present day data
        presentDayData = {}
        for i in data[0]:
            if str(data[0][i][0]) != 'nan':
                presentDayData[i] = data[0][i][0]


        finalOut = {
            'code' : 200,
            'response' : {
                "predicted_data" : predictions,
                "presentDayData" : presentDayData,
                "city_name" : data[1],
                "city_station" : data[2],
                "country_code" : data[3],
                # "plotImages" : plotImages
            }
        }

    else:
        finalOut = {
            'code' : 404
        }

    predictionsDataframe = pd.DataFrame(finalOut)

    predictionsDataframe.to_csv('Data/predictions_data.csv', index=False)
    return finalOut

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
)

@app.get("/hello", tags=['ROOT'])
async def root():
    # return JSONResponse()
    json_compatible_item_data = jsonable_encoder({"message": "Hello World"})
    return JSONResponse(content=json_compatible_item_data)

@app.get('/data/{city}')
async def data(city:str):
    data = getOnlyCityData(city_name=city)

    print(data)
    return data

@app.get('/predictions/{city}/{model}')
async def prophetData(city: str, model: str):
    print('-----------------------')
    city_data = requests.get(f'http://127.0.0.1:8000/data/{city}')
    print('-----------------------')

    string_city_data = city_data.content.decode('utf-8')

    # Step 2: Parse the string into a dictionary
    dict_city_data = json.loads(string_city_data)

    data = pd.DataFrame(dict_city_data)
    data = pd.to_datetime(data.index)
    models = Sktime_forecast(data)

    return JSONResponse(models.getAQIForecasts(model=model))

@app.get('/city/{city}')
async def city(city:str):
    # o = AQIData()
    city_data = getCityData(city_name=city)
    #get the predictions
    # predictions = forecaster.getForecastData(data=hist)
    
    return JSONResponse(city_data)

@app.get('/test/{text}')
async def test(text:str):
    print(text)
    return JSONResponse(
        {
            'text' : text
        }
    )