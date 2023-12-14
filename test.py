from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.fbprophet import Prophet
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas
import requests
from datetime import datetime, timedelta
from sseclient import SSEClient
import json
import js2py


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

    def parse_incoming_result(self, json_object: dict) -> pandas.DataFrame:
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

            series = pandas.Series(values, index=dates)
            result_dict[pollutant_name] = series

        FRAME = pandas.DataFrame(result_dict)
        return FRAME

        
    def get_results_from_backend(self, city_id: int):
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


    def get_data_from_id(self, city_id: int) -> pandas.DataFrame:
        backend_data = self.get_results_from_backend(city_id)
        result = pandas.concat([self.parse_incoming_result(data) for data in backend_data])
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
            complete_days = pandas.date_range(
                result.index.min(), result.index.max(), freq="D"
            )
            result = result.reindex(complete_days, fill_value=None)

            # Arrange to make most recent appear on top of DataFrame
            result = result.sort_index(ascending=False, na_position="last")

        return result

    
    def get_city_station_options(self, city: str) -> pandas.DataFrame:
        """Get available stations for a given city
        Args:
            city (str): Name of a city.

        Returns:
            pandas.DataFrame: Table of stations and their relevant information.
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

        return pandas.DataFrame(
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
    ) -> pandas.DataFrame:
        """Get historical air quality data for a city

        Args:
            city (str): Name of the city. If given, the argument must be named.
            city_id (int): City ID. If given, the argument must be named.
                If not given, city argument must not be None.

        Returns:
            pandas.DataFrame: The dataframe containing the data.
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


def sktime_forecast(dataset, horizon=30, forecaster=Prophet(yearly_seasonality=True, weekly_seasonality=True), validation=False, confidence=0.9, frequency="D"):
    """Loop over a time series dataframe, train an sktime forecasting model, and visualize the results.

    Args:
        dataset (pd.DataFrame): Input time series DataFrame with datetime index
        horizon (int): Forecast horizon
        forecaster (sktime.forecasting): Configured forecaster
        validation (bool, optional): . Defaults to False.
        confidence (float, optional): Confidence level. Defaults to 0.9.
        frequency (str, optional): . Defaults to "D".
    """
    # Adjust frequency
    forecast_df = dataset.resample(rule=frequency).sum()
    # Interpolate missing periods (if any)
    forecast_df = forecast_df.interpolate(method="time")

    all_parameters_values = {}
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
                pandas.date_range(str(present_date), periods=horizon, freq=frequency),
                is_relative=False,
            )

            y_pred = forecaster.predict(fh)
            ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")
            # mae = np.nan

        # Visualize results
        # plt.plot(
        #     df.tail(horizon * 3),
        #     label="Actual",
        #     color="black",
        # )
        # plt.gca().fill_between(
        #     ci.index, (ci.iloc[:, 0]), (ci.iloc[:, 1]), color="b", alpha=0.1
        # )
        # # print(y_pred)
        # plt.plot(y_pred, label="Predicted")
        # plt.xticks(rotation=45, ha='right')
        # # plt.title(
        # #     f"{horizon} day forecast for {col} (mae: {round(mae, 2)}, confidence: {confidence*100}%)"
        # # )
        # plt.ylim(bottom=0)
        # # plt.legend()
        # plt.grid(False)
        # plt.show()
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
    
    return predicted_data


def getCityData(city_name):
    o = AQIData()
    # dataset = o.get_historical_data(city="New York")
    # forecaster = AutoARIMA(sp=1, suppress_warnings=True)
    forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    # forecaster = ThetaForecaster(sp=12)
    data = o.get_historical_data(city=city_name)
    
    finalOut = {}
    if data != 404:

        dates = [i for i in data[0].index]
        for i in range(len(dates)):
            if(dates[i].date() > datetime.now().date()):
                data[0].drop(index=dates[i].date(), inplace=True)

        dataset = data[0]
        dataset = dataset.dropna()

        #saving the file locally without index
        # dataset.to_csv(f"Data/{city_name}_data.csv", index=False)

        #reading the file while parsing the dates
        # dataset = pd.read_csv(f"Data/{city_name}_data.csv", parse_dates=[0], index_col=[0])
        # print(dataset)
        #remove future dates in the dateset

        predicted_data = sktime_forecast(dataset=dataset,forecaster=forecaster, horizon=30, validation=False)


        #for present day data
        presentDayData = {}
        for i in data[0]:
            if str(data[0][i][0]) != 'nan':
                presentDayData[i] = data[0][i][0]


        finalOut = {
            'code' : 200,
            'response' : {
                "predicted_data" : predicted_data,
                "presentDayData" : presentDayData,
                "city_name" : data[1],
                "city_station" : data[2],
                "country_code" : data[3]
            }
        }

    else:
        finalOut = {
            'code' : 404
        }

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

@app.get('/city/{city}')
async def city(city:str):
    o = AQIData()
    hist = getCityData(city_name=city)
    #get the predictions 
    # predictions = forecaster.getForecastData(data=hist)

    return JSONResponse(hist)