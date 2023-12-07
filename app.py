
from fastapi import FastAPI

# from sktime.forecasting.base import ForecastingHorizon
# from sktime.forecasting.model_selection import temporal_train_test_split
# from sktime.forecasting.theta import ThetaForecaster
# from sktime.forecasting.naive import NaiveForecaster

# from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
import pandas as pd
import sys
# from sktime.forecasting.arima import AutoARIMA
# from sktime.forecasting.fbprophet import Prophet
import numpy as np
from datetime import timedelta

sys.path.append("..")
# import src.utility.plot_settings



# dataset = pd.read_csv("Data/TempNewDelhiData.csv", parse_dates=[0], index_col=[0])


# def sktime_forecast(dataset, horizon=30, forecaster=Prophet(yearly_seasonality=True, weekly_seasonality=True), validation=False, confidence=0.9, frequency="D"):
#     """Loop over a time series dataframe, train an sktime forecasting model, and visualize the results.

#     Args:
#         dataset (pd.DataFrame): Input time series DataFrame with datetime index
#         horizon (int): Forecast horizon
#         forecaster (sktime.forecasting): Configured forecaster
#         validation (bool, optional): . Defaults to False.
#         confidence (float, optional): Confidence level. Defaults to 0.9.
#         frequency (str, optional): . Defaults to "D".
#     """
#     # Adjust frequency
#     forecast_df = dataset.resample(rule=frequency).sum()
#     # Interpolate missing periods (if any)
#     forecast_df = forecast_df.interpolate(method="time")

#     all_parameters_values = {}
#     for col in dataset.columns:
#         # Use train/test split to validate forecaster
#         if validation:
#             df = forecast_df[col]

#             y_train = df[:-horizon]
#             y_test = df.tail(horizon)

#             forecaster.fit(y_train)
#             fh = ForecastingHorizon(y_test.index, is_relative=False)
#             y_pred = forecaster.predict(fh)
#             ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")
#             y_true = df.tail(horizon)

#             # mae = mean_absolute_error(y_true, y_pred)

#         # Make predictions beyond the dataset
#         if not validation:
#             df = forecast_df[col].dropna()
          
#             forecaster.fit(df)

#             #for present date            
#             present_date = datetime.now().date()
#             #to start predictions from tomorrow
#             present_date = str(present_date + timedelta(days=1)).split(' ')[0]
#             fh = ForecastingHorizon(
#                 pd.date_range(str(present_date), periods=horizon, freq=frequency),
#                 is_relative=False,
#             )

#             y_pred = forecaster.predict(fh)
#             ci = forecaster.predict_interval(fh, coverage=confidence).astype("float")
#             # mae = np.nan

#         # Visualize results
#         # plt.plot(
#         #     df.tail(horizon * 3),
#         #     label="Actual",
#         #     color="black",
#         # )
#         # plt.gca().fill_between(
#         #     ci.index, (ci.iloc[:, 0]), (ci.iloc[:, 1]), color="b", alpha=0.1
#         # )
#         # # print(y_pred)
#         # plt.plot(y_pred, label="Predicted")
#         # plt.xticks(rotation=45, ha='right')
#         # # plt.title(
#         # #     f"{horizon} day forecast for {col} (mae: {round(mae, 2)}, confidence: {confidence*100}%)"
#         # # )
#         # plt.ylim(bottom=0)
#         # # plt.legend()
#         # plt.grid(False)
#         # plt.show()
#         # print("Mean Absolute Error : ", mae)

#         # try :
#         #     temp = all_parameters_values['date']
#         # except:
#         #     all_parameters_values['date'] = [i.strftime("%d-%m-%Y") for i in fh]

#         all_parameters_values[col] = y_pred.values
    

    
#     dates = [i.strftime("%d-%m-%Y") for i in fh]

#     predicted_data = {}
#     for date in range(len(dates)):
#         temp = {}
#         for param in all_parameters_values:
#             temp[param] = all_parameters_values[param][date]
#         predicted_data[dates[date]] = temp
    
#     return predicted_data

# # def getPredictedData():





import json
from datetime import datetime
from typing import Any, Dict, List

import js2py
import pandas
import requests
from sseclient import SSEClient

JS_FUNCS: str = """
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
_context = js2py.EvalJs()
_context.execute(JS_FUNCS)


def get_results_from_backend(city_id: int) -> List[Dict[str, Any]]:
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


def parse_incoming_result(json_object: dict) -> pandas.DataFrame:
    # Run JS code
    # Function is defined within JS code above
    # Convert result to Python dict afterwards
    OUTPUT = _context.gatekeep_convert_date_object_to_unix_seconds(
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


def get_data_from_id(city_id: int) -> pandas.DataFrame:
    backend_data = get_results_from_backend(city_id)
    result = pandas.concat([parse_incoming_result(data) for data in backend_data])
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




"""Ozon3 module for the Ozon3 package.

This module contains the main Ozon3 class, which is used for all live-data
collection done by the Ozon3 package.

This module should be used only as a part of the Ozon3 package, and should not
be run directly.

Attributes (module level):
    CALLS (int=1000): The number of calls per second allowed by the WAQI API is 1000.
    RATE_LIMIT (int=1): The time period in seconds for the max number of calls is
        1 second.
"""
import itertools
import json
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy
import pandas
import requests
from ratelimit import limits, sleep_and_retry

# from .historical._reverse_engineered import get_data_from_id
# from .urls import URLs

# 1000 calls per second is the limit allowed by API
CALLS: int = 1000
RATE_LIMIT: int = 1


def _as_float(x: Any) -> float:
    """Convert x into a float. If unable, convert into numpy.nan instead.

    Naming and functionality inspired by R function as.numeric()"""
    try:
        return float(x)
    except (TypeError, ValueError):
        return numpy.nan
class URLs:
    """Class that contains the endpoint urls for the WAQI API

    This class should not be instantiated. It only contains class level attributes,
    and no methods at all. It is a static dataclass.

    Attributes:
        search_aqi_url (str): The endpoint used for retrieving air quality data.
        find_stations_url (str): The endpoint used for
         retrieving a collection of air quality measuring stations.
        find_coordinates_url (str): The endpoint used for
         retrieving geographical information
    """

    # Base API endpoint.
    _base_url: str = "https://api.waqi.info/"

    # For air quality data search by location.
    search_aqi_url: str = f"{_base_url}feed/"

    # For search for air quality measuring stations in area.
    find_stations_url: str = f"{_base_url}search/"

    # For Map Queries
    find_coordinates_url: str = f"{_base_url}map/"

class Ozon3:
    """Primary class for Ozon3 API

    This class contains all the methods used for data collection.
    This class should be instantiated, and methods should be called from the
    instance.

    Attributes:
        token (str): The private API token for the WAQI API service.
    """

    _search_aqi_url: str = URLs.search_aqi_url
    _find_stations_url: str = URLs.find_stations_url
    _default_params: List[str] = [
        "aqi",
        "pm2.5",
        "pm10",
        "o3",
        "co",
        "no2",
        "so2",
        "dew",
        "h",
        "p",
        "t",
        "w",
        "wg",
    ]

    def __init__(
        self, token: str = "", output_path: str = ".", file_name: str = "air_quality"
    ):
        """Initialises the class instance and sets the API token value

        Args:
            token (str): The users private API token for the WAQI API.
        """
        self.token: str = token
        self._check_token_validity()

    def _check_token_validity(self) -> None:
        """Check if the token is valid"""
        test_city: str = "london"
        r = self._make_api_request(
            f"{self._search_aqi_url}/{test_city}/?token={self.token}"
        )

        self._check_status_code(r)
        if json.loads(r.content)["status"] != "ok":
            warnings.warn("Token may be invalid!")

    @sleep_and_retry
    @limits(calls=CALLS, period=RATE_LIMIT)
    def _make_api_request(self, url: str) -> requests.Response:
        """Make an API request

        Args:
            url (str): The url to make the request to.

        Returns:
            requests.Response: The response from the API.
        """
        r = requests.get(url)
        return r

    def _check_status_code(self, r: requests.Response) -> None:
        """Check the status code of the response"""
        if r.status_code == 200:
            pass
        elif r.status_code == 401:
            raise Exception("Unauthorized!")
        elif r.status_code == 404:
            raise Exception("Not Found!")
        elif r.status_code == 500:
            raise Exception("Internal Server Error!")
        else:
            raise Exception(f"Error! Code {r.status_code}")

    def reset_token(self, token: str) -> None:
        """Use this method to set your API token

        Args:
            token (str): The new API token.
        """
        self.token = token
        self._check_token_validity()

    def _extract_live_data(self, data_obj: Any) -> Dict[str, Union[str, float]]:
        """Extract live AQI data from API response's 'data' part.

        Args:
            data_obj (JSON object returned by json.loads): The 'data' part from
                the API's response.

        Returns:
            dict: Dictionary containing the data.
        """

        # This dict will become a single row of data for the dataframe.
        row: Dict[str, Union[str, float]] = {}

        # City column can be added back later by the caller method.
        row["city"] = numpy.nan
        row["latitude"] = data_obj["city"]["geo"][0]
        row["longitude"] = data_obj["city"]["geo"][1]
        row["station"] = data_obj["city"]["name"]
        row["dominant_pollutant"] = data_obj["dominentpol"]
        if data_obj["dominentpol"] == "pm25":
            # Ensures that pm2.5 is correctly labeled.
            row["dominant_pollutant"] = "pm2.5"
        row["timestamp"] = data_obj["time"]["s"]
        row["timestamp_timezone"] = data_obj["time"]["tz"]

        for param in self._default_params:
            try:
                if param == "aqi":
                    # This is in different part of JSON object.
                    row["aqi"] = _as_float(data_obj["aqi"])
                    # This adds AQI_meaning and AQI_health_implications data.
                    (
                        row["AQI_meaning"],
                        row["AQI_health_implications"],
                    ) = self._AQI_meaning(_as_float(data_obj["aqi"]))
                elif param == "pm2.5":
                    # To ensure that pm2.5 data is labelled correctly.
                    row["pm2.5"] = _as_float(data_obj["iaqi"]["pm25"]["v"])
                else:
                    row[param] = _as_float(data_obj["iaqi"][param]["v"])
            except KeyError:
                # Gets triggered if the parameter is not provided by station.
                row[param] = numpy.nan

        return row

    def _extract_forecast_data(self, data_obj: Any) -> pandas.DataFrame:
        """Extract forecast data from API response's 'data' part.

        Args:
            data_obj (JSON object returned by json.loads): The 'data' part from
                the API's response.

        Returns:
            pandas.DataFrame: A dataframe containing the data.
        """
        forecast = data_obj["forecast"]["daily"]
        dict_of_frames = {}
        for pol, lst in forecast.items():
            dict_of_frames[pol] = pandas.DataFrame(lst).set_index("day")

        df = pandas.concat(dict_of_frames, axis=1)

        # Convert to numeric while making non-numerics nan,
        # and then convert to float, just in case there's int
        df = df.apply(lambda x: pandas.to_numeric(x, errors="coerce")).astype(float)
        df.index = pandas.to_datetime(df.index)
        df = df.reset_index().rename(columns={"day": "date"})
        return df

    def _check_and_get_data_obj(
        self, r: requests.Response, **check_debug_info
    ) -> Union[dict, List[dict]]:
        """Get data object from API response and throw error if any is encouuntered

        Args:
            r (requests.Response): Response object from API request.
            **check_debug_info: Any debug info that can help make
                exceptions in this method more informative. Give this argument in
                format of e.g. `city="London"` to allow exceptions that can take
                city names to show it instead of just generic exception message.

        Returns:
            Union[dict, List[dict]]: The data object i.e. the `data` part of the
                API response, in dictionary or list format (already JSON-ified).

        """
        self._check_status_code(r)

        response = json.loads(r.content)
        status = response.get("status")
        data = response.get("data")

        if status == "ok":
            if isinstance(data, dict) or isinstance(data, list):
                # Only return data if status is ok and data is either dict or list.
                # Otherwise it gets to exception raisers below.
                return data

        if isinstance(data, str):
            if "Unknown station" in data:
                # Usually happens when WAQI does not have a station
                # for the searched city name.

                # Check if a city name is provided so that user can get
                # better exception message to aid them debug their program
                city = check_debug_info.get("city")
                city_info = f'\ncity: "{city}"' if city is not None else ""

                raise Exception(
                    "There is no known AQI station for the given query." + city_info
                )

            if "Invalid geo position" in data:
                # Usually happens when WAQI can't parse the given
                # lat-lon coordinate.

                # data is fortunately already informative
                raise Exception(f"{data}")

            if "Invalid key" in data:
                raise Exception("Your API token is invalid.")

            # Unlikely since rate limiter is already used,
            # but included anyway for completeness.
            if "Over quota" in data:
                raise Exception("Too many requests within short time.")

        # Catch-all exception for other not yet known cases
        raise Exception(f"Can't parse the returned response:\n{response}")

    def _AQI_meaning(self, aqi: float) -> Tuple[str, str]:
        """Retrieve AQI meaning and health implications

        Args:
            aqi (float): Air Quality Index (AQI) value.

        Returns:
            Tuple[str, str]: The meaning and health implication of the AQI value.
        """

        if 0 <= aqi <= 50:
            AQI_meaning = "Good"
            AQI_health_implications = (
                "Air quality is considered satisfactory, "
                "and air pollution poses little or no risk"
            )
        elif 51 <= aqi <= 100:
            AQI_meaning = "Moderate"
            AQI_health_implications = (
                "Air quality is acceptable; however, for some pollutants "
                "there may be a moderate health concern for a very small "
                "number of people who are unusually sensitive to air pollution."
            )
        elif 101 <= aqi <= 150:
            AQI_meaning = "Unhealthy for sensitive group"
            AQI_health_implications = (
                "Members of sensitive groups may experience health effects. "
                "The general public is not likely to be affected."
            )
        elif 151 <= aqi <= 200:
            AQI_meaning = "Unhealthy"
            AQI_health_implications = (
                "Everyone may begin to experience health effects; members of "
                "sensitive groups may experience more serious health effects."
            )
        elif 201 <= aqi <= 300:
            AQI_meaning = "Very Unhealthy"
            AQI_health_implications = (
                "Health warnings of emergency conditions. "
                "The entire population is more likely to be affected."
            )
        elif 301 <= aqi <= 500:
            AQI_meaning = "Hazardous"
            AQI_health_implications = (
                "Health alert: everyone may experience more serious health effects."
            )
        else:
            AQI_meaning = "Invalid AQI value"
            AQI_health_implications = "Invalid AQI value"

        return AQI_meaning, AQI_health_implications

    def _locate_all_coordinates(
        self, lower_bound: Tuple[float, float], upper_bound: Tuple[float, float]
    ) -> List[Tuple]:
        """Get all locations between two pair of coordinates

        Args:
            lower_bound (tuple): start location
            upper_bound (tuple): end location

        Returns:
           list: a list of all coordinates located between lower_bound and
               upper_bound.
        """

        coordinates_flattened: List[float] = list(
            itertools.chain(lower_bound, upper_bound)
        )
        latlng: str = ",".join(map(str, coordinates_flattened))
        response = self._make_api_request(
            f"{URLs.find_coordinates_url}bounds/?token={self.token}&latlng={latlng}"
        )

        data = self._check_and_get_data_obj(response)

        coordinates: List[Tuple] = [
            (element["lat"], element["lon"]) for element in data
        ]
        return coordinates

    def get_coordinate_air(
        self,
        lat: float,
        lon: float,
        df: pandas.DataFrame = pandas.DataFrame(),
    ) -> pandas.DataFrame:
        """Get a location's air quality data by latitude and longitude

        Args:
            lat (float): Latitude
            lon (float): Longitude
            df (pandas.DataFrame, optional): An existing dataframe to
                append the data to.

        Returns:
            pandas.DataFrame: The dataframe containing the data.
        """
        r = self._make_api_request(
            f"{self._search_aqi_url}/geo:{lat};{lon}/?token={self.token}"
        )
        data_obj = self._check_and_get_data_obj(r)

        row = self._extract_live_data(data_obj)
        df = pandas.concat([df, pandas.DataFrame([row])], ignore_index=True)
        return df

    def get_city_air(
        self,
        city: str,
        df: pandas.DataFrame = pandas.DataFrame(),
    ) -> pandas.DataFrame:
        """Get a city's air quality data

        Args:
            city (str): The city to get data for.
            df (pandas.DataFrame, optional): An existing dataframe to
                append the data to.

        Returns:
            pandas.DataFrame: The dataframe containing the data.
        """
        r = self._make_api_request(f"{self._search_aqi_url}/{city}/?token={self.token}")
        data_obj = self._check_and_get_data_obj(r, city=city)  # City is for traceback

        row = self._extract_live_data(data_obj)
        row["city"] = city

        df = pandas.concat([df, pandas.DataFrame([row])], ignore_index=True)
        return df

    def get_multiple_coordinate_air(
        self,
        locations: List[Tuple],
        df: pandas.DataFrame = pandas.DataFrame(),
    ) -> pandas.DataFrame:
        """Get multiple locations air quality data

        Args:
            locations (list): A list of pair (latitude,longitude) to get data for.
            df (pandas.DataFrame, optional): An existing dataframe to
                append the data to.

        Returns:
            pandas.DataFrame: The dataframe containing the data.
        """
        for loc in locations:
            try:
                # This just makes sure that it's always a returns a pandas.DataFrame.
                # Makes mypy happy.
                df = pandas.DataFrame(self.get_coordinate_air(loc[0], loc[1], df=df))
            except Exception:
                # NOTE: If we have custom exception we can catch it instead.
                empty_row = pandas.DataFrame(
                    {"latitude": [_as_float(loc[0])], "longitude": [_as_float(loc[1])]}
                )
                df = pandas.concat([df, empty_row], ignore_index=True)

        df.reset_index(inplace=True, drop=True)
        return df

    def get_range_coordinates_air(
        self,
        lower_bound: Tuple[float, float],
        upper_bound: Tuple[float, float],
        df: pandas.DataFrame = pandas.DataFrame(),
    ) -> pandas.DataFrame:
        """Get aqi data for range of coordinates b/w lower_bound and upper_bound

        Args:
            lower_bound (tuple): start coordinate
            upper_bound (tuple): end coordinate
            df (pandas.DataFrame, optional): An existing dataframe to
                append the data to.

        Returns:
            pandas.DataFrame: The dataframe containing the data.
        """
        locations = self._locate_all_coordinates(
            lower_bound=lower_bound, upper_bound=upper_bound
        )
        return self.get_multiple_coordinate_air(locations, df=df)

    def get_multiple_city_air(
        self,
        cities: List[str],
        df: pandas.DataFrame = pandas.DataFrame(),
    ) -> pandas.DataFrame:
        """Get multiple cities' air quality data

        Args:
            cities (list): A list of cities to get data for.
            df (pandas.DataFrame, optional): An existing dataframe to
                append the data to.

        Returns:
            pandas.DataFrame: The dataframe containing the data.
        """
        for city in cities:
            try:
                # This just makes sure that it's always a returns a pandas.DataFrame.
                # Makes mypy happy.
                df = pandas.DataFrame(self.get_city_air(city=city, df=df))
            except Exception:
                # NOTE: If we have custom exception we can catch it instead.
                empty_row = pandas.DataFrame({"city": [city]})
                df = pandas.concat([df, empty_row], ignore_index=True)

        df.reset_index(inplace=True, drop=True)
        return df

    def get_specific_parameter(
        self,
        city: str,
        air_param: str = "",
    ) -> float:
        """Get specific parameter as a float

        Args:
            city (string): A city to get the data for
            air_param (string): A string containing the specified air quality parameter.
                Choose from the following values:
                ["aqi", "pm2.5", "pm10", "o3", "co", "no2", "so2", "dew", "h",
                 "p", "t", "w", "wg"]
                Gets all parameters by default.

        Returns:
            float: Value of the specified parameter for the given city.
        """
        r = self._make_api_request(f"{self._search_aqi_url}/{city}/?token={self.token}")
        data_obj = self._check_and_get_data_obj(r)

        row = self._extract_live_data(data_obj)

        try:
            result = _as_float(row[air_param])
        except KeyError:
            raise Exception(
                f'Missing air quality parameter "{air_param}"\n'
                'Try another air quality parameters: "aqi", "no2", or "co"'
            )

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

            # warnings.warn(
            #     f'city_id was not supplied. Searching for "{city}" yields '
            #     f'city ID {city_id} with station name "{station_name}", '
            #     f'with country code "{country_code}". '
            #     "Ozon3 will return air quality data from that station. "
            #     "If you know this is not the correct city you intended, "
            #     "you can use get_city_station_options method first to "
            #     "identify the correct city ID."
            # )
        else:
            if city is not None:
                warnings.warn(
                    "Both arguments city and city_id were supplied. "
                    "Only city_id will be used. city argument will be ignored."
                )

        df = get_data_from_id(city_id)
        if "pm25" in df.columns:
            # This ensures that pm25 data is labelled correctly.
            df.rename(columns={"pm25": "pm2.5"}, inplace=True)

        # Reset date index and rename the column appropriately
        # df = df.reset_index().rename(columns={"index": "date"})
        # print(df)

        return [df ,city , station_name, country_code]

    def get_city_forecast(
        self,
        city: str,
        df: pandas.DataFrame = pandas.DataFrame(),
    ) -> pandas.DataFrame:
        """Get a city's air quality forecast

        Args:
            city (str): The city to get data for.
            df (pandas.DataFrame, optional): An existing dataframe to
                append the data to.

        Returns:
            pandas.DataFrame: The dataframe containing the data.
        """
        r = self._make_api_request(f"{self._search_aqi_url}/{city}/?token={self.token}")
        data_obj = self._check_and_get_data_obj(r)

        df = self._extract_forecast_data(data_obj)
        if "pm25" in df.columns:
            # This ensures that pm25 data is labelled correctly.
            df.rename(columns={"pm25": "pm2.5"}, inplace=True)

        return df


# def getCityData(city_name):
#     o = Ozon3('a36388df93e27e7fb00282d007eae2e68c561a61')
#     # dataset = o.get_historical_data(city="New York")
#     # forecaster = AutoARIMA(sp=1, suppress_warnings=True)
#     forecaster = Prophet(yearly_seasonality=True, weekly_seasonality=True)
#     # forecaster = ThetaForecaster(sp=12)
#     data = o.get_historical_data(city=city_name)
    
#     finalOut = {}
#     if data != 404:

#         dates = [i for i in data[0].index]
#         for i in range(len(dates)):
#             if(dates[i].date() > datetime.now().date()):
#                 data[0].drop(index=dates[i].date(), inplace=True)

#         dataset = data[0]
#         dataset = dataset.dropna()

#         #saving the file locally without index
#         # dataset.to_csv(f"Data/{city_name}_data.csv", index=False)

#         #reading the file while parsing the dates
#         # dataset = pd.read_csv(f"Data/{city_name}_data.csv", parse_dates=[0], index_col=[0])
#         # print(dataset)
#         #remove future dates in the dateset

#         predicted_data = sktime_forecast(dataset=dataset,forecaster=forecaster, horizon=30, validation=False)


#         #for present day data
#         presentDayData = {}
#         for i in data[0]:
#             if str(data[0][i][0]) != 'nan':
#                 presentDayData[i] = data[0][i][0]


#         finalOut = {
#             'code' : 200,
#             'response' : {
#                 "predicted_data" : predicted_data,
#                 "presentDayData" : presentDayData,
#                 "city_name" : data[1],
#                 "city_station" : data[2],
#                 "country_code" : data[3]
#             }
#         }

#     else:
#         finalOut = {
#             'code' : 404
#         }

#     return finalOut

app = FastAPI()

@app.get("/", tags=['ROOT'])
async def root():
    return {"message": "Hello World"}

@app.get('/city')
async def city():
    o = Ozon3('a36388df93e27e7fb00282d007eae2e68c561a61')
    hist = o.get_historical_data(city_name='Delhi')
    #get the predictions 
    # predictions = forecaster.getForecastData(data=hist)

    return {
        'data' : hist,
        'response' : 200
    }