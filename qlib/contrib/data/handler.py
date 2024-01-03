# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class Alpha360(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    @staticmethod
    def get_feature_config():
        # NOTE:
        # Alpha360 tries to provide a dataset with original price data
        # the original price data includes the prices and volume in the last 60 days.
        # To make it easier to learn models from this dataset, all the prices and volume
        # are normalized by the latest price and volume data ( dividing by $close, $volume)
        # So the latest normalized $close will be 1 (with name CLOSE0), the latest normalized $volume will be 1 (with name VOLUME0)
        # If further normalization are executed (e.g. centralization),  CLOSE0 and VOLUME0 will be 0.
        fields = []
        names = []

        for i in range(59, 0, -1):
            fields += ["Ref($close, %d)/$close" % i]
            names += ["CLOSE%d" % i]
        fields += ["$close/$close"]
        names += ["CLOSE0"]
        for i in range(59, 0, -1):
            fields += ["Ref($open, %d)/$close" % i]
            names += ["OPEN%d" % i]
        fields += ["$open/$close"]
        names += ["OPEN0"]
        for i in range(59, 0, -1):
            fields += ["Ref($high, %d)/$close" % i]
            names += ["HIGH%d" % i]
        fields += ["$high/$close"]
        names += ["HIGH0"]
        for i in range(59, 0, -1):
            fields += ["Ref($low, %d)/$close" % i]
            names += ["LOW%d" % i]
        fields += ["$low/$close"]
        names += ["LOW0"]
        for i in range(59, 0, -1):
            fields += ["Ref($vwap, %d)/$close" % i]
            names += ["VWAP%d" % i]
        fields += ["$vwap/$close"]
        names += ["VWAP0"]
        for i in range(59, 0, -1):
            fields += ["Ref($volume, %d)/($volume+1e-12)" % i]
            names += ["VOLUME%d" % i]
        fields += ["$volume/($volume+1e-12)"]
        names += ["VOLUME0"]

        return fields, names


class Alpha360vwap(Alpha360):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


class Alpha158(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        
        ## YUYUAN WANG: modify feature_config file ###
        ## add ADARNN with Alpha158 all fields (in **feature_config** item)
        feature_config = {
            'window.size' : 5, # window size by default
            'kbar': {'windows': [0,1,2,3,4],
                     'features': []}, # whether to use some hard-code kbar features
            # whether to use raw price features
            'price': {'windows': [0,1,2,3,4],
                      'features': ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]}, # which price field (ratio) to use
            # whether to use raw volume (ratio) features 
            'volume': {'windows': [0,1,2,3,4]},
             # whether to use rolling operator based features
            # window should not include 0 !!
            'rolling': {'windows': [5, 10, 20, 30, 60],
                        'features': []},
             # additional features to be used; must have the same window size
            # MUST be a price object!
            'additional': {'windows': [0,1,2,3,4],
                           'features': ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]}       
        },
        
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(feature_config),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )

    def get_feature_config(self, feature_config):
        return self.parse_config_to_fields(feature_config)

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]

    @staticmethod
    def parse_config_to_fields(config):
        """create factors from config

        config = {
            'window.size' : 5, # window size by default
            'kbar': {'windows': [0,1,2,3,4],
                     'features': []}, # whether to use some hard-code kbar features
            # whether to use raw price features
            'price': {'windows': [0,1,2,3,4],
                      'features': ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]}, # which price field (ratio) to use
            # whether to use raw volume (ratio) features (Yes/No)
            'volume': {'windows': [0,1,2,3,4]},
             # whether to use rolling operator based features
            # window should not include 0 !!
            'rolling': {'windows': [5, 10, 20, 30, 60],
                        'features': []},
             # additional features to be used; must have the same window size
             # MUST be a price object!
            'additional': {'windows': [0,1,2,3,4],
                           'features': ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]}       
        }
        """
        fields = []
        names = []
        if "kbar" in config: # finshed
            # kbar all features
            windows = config["kbar"].get("windows", [0,1,2,3,4])
            kbar_pool = {
                "KMID": "($close-$open)/$open",
                "KLEN": "($high-$low)/$open",
                "KMID2": "($close-$open)/($high-$low+1e-12)",
                "KUP": "($high-Greater($open, $close))/$open",
                "KUP2": "($high-Greater($open, $close))/($high-$low+1e-12)",
                "KLOW": "(Less($open, $close)-$low)/$open",
                "KLOW2": "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "KSFT": "(2*$close-$high-$low)/$open",
                "KSFT2": "(2*$close-$high-$low)/($high-$low+1e-12)"}
            for _ in config['kbar']['features']:
                names += [_ + '_' + str(d) for d in windows]
                if _ == "KMID": fields += ["(Ref($close, %d)-Ref($open, %d))/Ref($open, %d)" % (d, d, d) if d != 0 else kbar_pool[_] for d in windows]   
                if _ == "KLEN": fields += ["(Ref($high, %d)-Ref($low, %d))/Ref($open, %d)" % (d, d, d) if d != 0 else kbar_pool[_] for d in windows]
                if _ == "KMID2": fields += ["(Ref($close, %d)-Ref($open, %d))/(Ref($high, %d)-Ref($low, %d)+1e-12)" % (d, d, d, d) if d != 0 else kbar_pool[_] for d in windows]
                if _ == "KUP": fields += ["(Ref($high, %d)-Greater(Ref($open, %d), Ref($close, %d)))/Ref($open, %d)" % (d, d, d, d) if d != 0 else kbar_pool[_] for d in windows] 
                if _ == "KUP2": fields += ["(Ref($high, %d)-Greater(Ref($open, %d), Ref($close, %d)))/(Ref($high, %d)-Ref($low, %d)+1e-12)" % (d, d, d, d, d) if d != 0 else kbar_pool[_] for d in windows]  
                if _ == "KLOW": fields += ["(Less(Ref($open, %d), Ref($close, %d)) - Ref($low, %d))/Ref($open, %d)" % (d, d, d, d) if d != 0 else kbar_pool[_] for d in windows] 
                if _ == "KLOW2": fields += ["(Less(Ref($open, %d), Ref($close, %d)) - Ref($low, %d))/(Ref($high, %d)-Ref($low, %d)+1e-12)" % (d, d, d, d, d) if d != 0 else kbar_pool[_] for d in windows]  
                if _ == "KSFT": fields += ["(2*Ref($close, %d) - Ref($high, %d) - Ref($low, %d))/Ref($open, %d)" % (d, d, d, d) if d != 0 else kbar_pool[_] for d in windows] 
                if _ == "KSFT2": fields += ["(2*Ref($close, %d) - Ref($high, %d) - Ref($low, %d))/(Ref($high, %d)-Ref($low, %d)+1e-12)" % (d, d, d, d, d) if d != 0 else kbar_pool[_] for d in windows]  
                

        if "price" in config: # begin
            windows = config['price'].get("windows", [0,1,2,3,4])
            feature = config["price"].get("features", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for field in feature: # MUST BE A ratio to the current close
                field = field.lower()
                fields += ["Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field for d in windows]
                names += [field.upper() + '_' + str(d) for d in windows]
        if "volume" in config: # volumn ratio
            windows = config['volume'].get("windows", [0,1,2,3,4])
            fields += ["Ref($volume, %d)/($volume+1e-12)" % d if d != 0 else "$volume/($volume+1e-12)" for d in windows]
            names += ["VOLUME" + '_' + str(d) for d in windows]
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("features", None)
            # `exclude` in dataset config unnecessary filed
            # `include` in dataset config necessary field

            def use(x):
                return include is None or x in include

            # Some factor ref: https://guorn.com/static/upload/file/3/134065454575605.pdf
            if use("ROC"):
                # https://www.investopedia.com/terms/r/rateofchange.asp
                # Rate of change, the price change in the past d days, divided by latest close price to remove unit
                fields += ["Ref($close, %d)/$close" % d for d in windows]
                names += ["ROC_%d" % d for d in windows]
            if use("MA"):
                # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
                # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
                fields += ["Mean($close, %d)/$close" % d for d in windows]
                names += ["MA_%d" % d for d in windows]
            if use("STD"):
                # The standard diviation of close price for the past d days, divided by latest close price to remove unit
                fields += ["Std($close, %d)/$close" % d for d in windows]
                names += ["STD_%d" % d for d in windows]
            if use("BETA"):
                # The rate of close price change in the past d days, divided by latest close price to remove unit
                # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
                fields += ["Slope($close, %d)/$close" % d for d in windows]
                names += ["BETA_%d" % d for d in windows]
            if use("RSQR"):
                # The R-sqaure value of linear regression for the past d days, represent the trend linear
                fields += ["Rsquare($close, %d)" % d for d in windows]
                names += ["RSQR_%d" % d for d in windows]
            if use("RESI"):
                # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
                fields += ["Resi($close, %d)/$close" % d for d in windows]
                names += ["RESI_%d" % d for d in windows]
            if use("MAX"):
                # The max price for past d days, divided by latest close price to remove unit
                fields += ["Max($high, %d)/$close" % d for d in windows]
                names += ["MAX_%d" % d for d in windows]
            if use("LOW"):
                # The low price for past d days, divided by latest close price to remove unit
                fields += ["Min($low, %d)/$close" % d for d in windows]
                names += ["MIN_%d" % d for d in windows]
            if use("QTLU"):
                # The 80% quantile of past d day's close price, divided by latest close price to remove unit
                # Used with MIN and MAX
                fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
                names += ["QTLU_%d" % d for d in windows]
            if use("QTLD"):
                # The 20% quantile of past d day's close price, divided by latest close price to remove unit
                fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
                names += ["QTLD_%d" % d for d in windows]
            if use("RANK"):
                # Get the percentile of current close price in past d day's close price.
                # Represent the current price level comparing to past N days, add additional information to moving average.
                fields += ["Rank($close, %d)" % d for d in windows]
                names += ["RANK_%d" % d for d in windows]
            if use("RSV"):
                # Represent the price position between upper and lower resistent price for past d days.
                fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
                names += ["RSV_%d" % d for d in windows]
            if use("IMAX"):
                # The number of days between current date and previous highest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
                names += ["IMAX_%d" % d for d in windows]
            if use("IMIN"):
                # The number of days between current date and previous lowest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
                names += ["IMIN_%d" % d for d in windows]
            if use("IMXD"):
                # The time period between previous lowest-price date occur after highest price date.
                # Large value suggest downward momemtum.
                fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
                names += ["IMXD_%d" % d for d in windows]
            if use("CORR"):
                # The correlation between absolute close price and log scaled trading volume
                fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
                names += ["CORR_%d" % d for d in windows]
            if use("CORD"):
                # The correlation between price change ratio and volume change ratio
                fields += ["Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows]
                names += ["CORD_%d" % d for d in windows]
            if use("CNTP"):
                # The percentage of days in past d days that price go up.
                fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTP_%d" % d for d in windows]
            if use("CNTN"):
                # The percentage of days in past d days that price go down.
                fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTN_%d" % d for d in windows]
            if use("CNTD"):
                # The diff between past up day and past down day
                fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
                names += ["CNTD_%d" % d for d in windows]
            if use("SUMP"):
                # The total gain / the absolute total price changed
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMP_%d" % d for d in windows]
            if use("SUMN"):
                # The total lose / the absolute total price changed
                # Can be derived from SUMP by SUMN = 1 - SUMP
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMN_%d" % d for d in windows]
            if use("SUMD"):
                # The diff ratio between total gain and total lose
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
                    "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["SUMD_%d" % d for d in windows]
            if use("VMA"):
                # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
                fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VMA_%d" % d for d in windows]
            if use("VSTD"):
                # The standard deviation for volume in past d days.
                fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VSTD_%d" % d for d in windows]
            if use("WVMA"):
                # The volume weighted price change volatility
                fields += [
                    "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["WVMA_%d" % d for d in windows]
            if use("VSUMP"):
                # The total volume increase / the absolute total volume changed
                fields += [
                    "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMP_%d" % d for d in windows]
            if use("VSUMN"):
                # The total volume increase / the absolute total volume changed
                # Can be derived from VSUMP by VSUMN = 1 - VSUMP
                fields += [
                    "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMN_%d" % d for d in windows]
            if use("VSUMD"):
                # The diff ratio between total volume increase and total volume decrease
                # RSI indicator for volume
                fields += [
                    "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
                    "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["VSUMD_%d" % d for d in windows]
        if 'additional' in config: # MUST be a price object!
            windows = config['additional'].get("windows", [0,1,2,3,4])
            feature = config["additional"].get("features", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for field in feature: # WITH UNITS!
                field = field.lower()
                fields += ["Ref($%s, %d)" % (field, d) if d != 0 else "$%s" % field for d in windows]
                names += [field.upper() + '_' + str(d) + 'pure' for d in windows]

        return fields, names


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]
