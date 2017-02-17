# Kaggle competition "rossmann-stores-sales"

options(warn = 0)

library (Matrix)
library (data.table)
library (zoo)
library (xgboost)
library (forecast)
library (glmnetUtils)
library (lubridate)
library (caret)
library (ComputeBackend)

config = list()

config$do.log          = F
config$do.preprocess1  = T
config$do.pipeline     = T
config$do.preprocess2  = T
config$do.submission   = T

config$single.mode     = T
config$bagging.mode    = F
config$tunning.mode    = F

config$data.tag  = '2'
config$model.tag = '4'
config$submit.tag = 'pomo'

config$debug.preprocess1 = F
config$debug.preprocess2 = F
config$debug.model       = F
config$single.mode.day   = 48
config$rolling.days      = 1:48
config$show.feature.importance = F

config$compute.backend = 'serial' # {serial, multicore, condor, pbs}
config$nr.cores = ifelse(config$compute.backend %in% c('condor', 'pbs'), 200, 8)
config$rng.seed = 123456789
config$nr.threads = detectCores(all.tests = F, logical = F) # for computation on this machine

config$package.dependencies = c('methods', 'Matrix', 'data.table', 'ComputeBackend', 'xgboost', 'forecast', 'lubridate', 'caret')
config$source.dependencies  = NULL
config$cluster.dependencies = NULL
config$cluster.requirements = 'FreeMemoryMB >= 7500' # we're kinda RAM intensive with this one...

config$datadir = '../../Data/Kaggle/rossmann-store-sales'
config$project.dir = system('pwd', intern = T)

config$remove.zero.sales  = T
config$recode.nas         = T
config$remove.outliers    = F
config$remove.xmas        = F
config$remove.holidays    = F
config$validation.scheme  = 'special' # { none, last, first, stoypy, special, random }
config$hide.validation    = F
config$recode.na.as       = -999
config$include.pcs        = 10
config$take.log           = T
config$calibration.scheme = 'best' # { none, best, 0.985, linear, affine, robust, spline, isoreg }
config$weights.sel        = 'special' # {uniform, step, linear, exponential, special} NOTE: only used when take.log == F
config$use.model          = 'xgb' # { xgb, ts, storewise } 

config$get.validation.idxs = function(config, train, test) {
  nr.test.days = max(test$Date) - min(test$Date) + 1
  train.dates = sort(unique(train$Date))
  
  if (config$validation.scheme == 'last') {
    last.train.date = max(train$Date) - nr.test.days
    valid.dates = train.dates[train.dates > last.train.date]
  } else if (config$validation.scheme == 'first') {
    first.train.date = min(train$Date) + nr.test.days
    valid.dates = train.dates[train.dates < first.train.date]
  } else if (config$validation.scheme == 'stoypy') {
    first.valid.date = min(test$Date) - years(1)
    last.valid.date  = max(test$Date) - years(1)
    valid.dates = train.dates[(train.dates >= first.valid.date) & (train.dates <= last.valid.date)]
  } else if (config$validation.scheme == 'special') {
    last.train.date = max(train$Date) - nr.test.days
    first.stoypy.date = min(test$Date) - years(1)
    last.stoypy.date  = max(test$Date) - years(1)
    valid.dates = train.dates[(as.numeric(train.dates) %% 2 == 0) & (((train.dates >= first.stoypy.date) & (train.dates <= last.stoypy.date)) | (train.dates > last.train.date))]
  } else if (config$validation.scheme == 'random') {
    if (0) {
      valid.dates = sample(train.dates, nr.test.days)
    } else {
      cat('NOTE: experimental smartass validation split\n')
      # TODO: deal with these issues
      # want to sample later dates more, somehow match promo
      # some stores don't appear in the testset, others are missing half a year in the training data..
      # the distribution of weekdays is almost uniform in both sets
      # there are almost no state holidays in the testset
      # there are more promo days in the testset relative to the trainset, possibly varying between sets for some stores...
      cand.dates = train[, Date[all(StateHoliday == '0')], by = Date]$V1 # days that are not holy
      cand.dates = cand.dates[month(cand.dates) > 1 & month(cand.dates) < 11] # days not in the xmas shopping spree
      daynames = c('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
      valid.dates = NULL
      for (wd in 1:7) {
        ntst = floor(nrow(test[Store == 1 & DayOfWeek == wd]) / 2)
        valid.dates = c(valid.dates, sample(cand.dates[weekdays(cand.dates) == daynames[wd]], ntst))
      }
      valid.dates = as.Date(valid.dates)
    }
  }

  # NOTE: some processing like PCA assumes that the split is done at the resolution of whole dates
  valid.idxs = train$Date %in% valid.dates
  return (valid.idxs)
}

config$preprocess1 = function(config) {
  # FIXME make more efficient use of data.table...
  
  #
  cat(date(), 'Loading data\n')
  #
  
  # Original Kaggle data
  train = fread(paste0(config$datadir, '/train.csv'), stringsAsFactors = T)
  test  = fread(paste0(config$datadir, '/test.csv' ), stringsAsFactors = T)
  store = fread(paste0(config$datadir, '/store.csv'), stringsAsFactors = T)
  
  # External data about store states (all external data was posted on the competition forum as instructed by the rules)
  store.states = fread(paste0(config$datadir, '/EXT_store_states.csv'), stringsAsFactors = T)
  states = fread(paste0(config$datadir, '/state_stats.csv'), stringsAsFactors = T)
  # I guess we can't tell NI from HB due to holidays being identical, so merge them... and hope for the best
  store.states[State == 'HB,NI', State := 'NI']
  states[, GDP := rank(GDP)] # might be useful as an ordered representation of state, or I could use the Sales data per state...
  states = states[-which(State == 'HB'), .(State, GDP)]
  setnames(states, 'GDP', 'rnkState')
  store.states = merge(store.states, states, by = 'State')
  store.states[, State := factor(State)]
  store = merge(store, store.states, by = 'Store')
  # may need these
  store[, imputedCompetitionDistance := ifelse(is.na(CompetitionDistance), 50e3, CompetitionDistance)]

  # Some initial essential stuff
  train[, Id        := -(nrow(train):1)]
  test [, Sales     := NA              ]
  test [, Customers := NA              ]
  train[, Date      := as.Date(Date)   ]
  test [, Date      := as.Date(Date)   ]

  #
  # Decide on a train/validation split, and hold out validation data so
  # that validation faithfully reflects the testset situation
  #
  
  valid.truth = NULL
  
  if (config$validation.scheme != 'none' && config$hide.validation) {
    cat('NOTE: Hiding data for validation scheme:', config$validation.scheme, '\n')
    valid.idxs = config$get.validation.idxs(config, train, test)
    valid.truth = train[valid.idxs, .(Id, Store, Date, Sales, Customers)]
  
    train[valid.idxs, Sales     := NA]
    train[valid.idxs, Customers := NA]
  }
  
  #
  cat(date(), 'Cleaning and processing\n')
  #
  
  # Missing Open in test should be 0 (as per admin commnet https://www.kaggle.com/c/rossmann-store-sales/forums/t/17048/putting-stores-on-the-map?page=2)
  test[is.na(Open), Open := 0]

  # Promo2 stuff per store
  store[, Promo2Since := as.Date(as.POSIXct(paste(store$Promo2SinceYear, store$Promo2SinceWeek, 1, sep = "-"), format = "%Y-%U-%u"))]
  store[, Promo2SinceYear := NULL]
  store[, Promo2SinceWeek := NULL]
  store[, imputedPromo2Since := ifelse(is.na(Promo2Since), median(Promo2Since, na.rm = T), Promo2Since)]
  
  # Competition stuff per store
  store[, CompetitionOpenSince := as.Date(as.POSIXct(paste(store$CompetitionOpenSinceYear, store$CompetitionOpenSinceMonth, 1, sep = "-"), format = "%Y-%m-%d"))]
  store[, CompetitionOpenSinceYear  := NULL]
  store[, CompetitionOpenSinceMonth := NULL]
  store[, imputedCompetitionOpenSince := ifelse(is.na(CompetitionOpenSince), median(CompetitionOpenSince, na.rm = T), CompetitionOpenSince)]
  
  # Merge train and test so that processing is identical
  data = rbind(train, test)

  # Date stuff
  data[, year      := as.integer(format(Date, '%y'))]
  data[, month     := as.integer(format(Date, '%m'))]
  data[, day       := as.integer(format(Date, '%d'))]
  data[, absday    := as.integer(Date)]
  data[, DayOfWeek := as.factor(DayOfWeek)]
  
  data[, xmas   := as.numeric(month == 12)]
  data[, winter := as.numeric(month %in% c(12, 1:3))]
  data[, spring := as.numeric(month %in% 5:6)]
  data[, summer := as.numeric(month %in% 7:9)]

  # Binary, so represent with an integer
  data[, SchoolHoliday := as.integer(SchoolHoliday)]
  
  # This leaks a bit, but worth it
  store[, intStoreType := as.integer(StoreType)]
  store[intStoreType == 2, intStoreType := -1] # I'm just trying to make it easier to see type 'b' sells more, assuming tree-based models, the actual coding doesn't matter
  store[, intAssortment := as.integer(Assortment)]
  store[intAssortment == 2, intAssortment := 4] # same comment
  
  #
  cat(date(), 'Merging store info\n')
  #
  
  # Merge store data
  data = merge(data, store, by = 'Store')

  # Is store open on Sundays?
  sun.stores = unique(data[DayOfWeek == 7 & Sales > 0, Store])
  data[, sunStore := 0]
  data[Store %in% sun.stores, sunStore := 1]
  
  # Promo2 stuff per day
  data[, inPromo2 := as.numeric(Date >= Promo2Since)]
  data[is.na(inPromo2), inPromo2 := 0]
  data[, newPromo2 := 0]
  data[PromoInterval == 'Feb,May,Aug,Nov'  & (month %in% c(2, 5, 8, 11)), newPromo2 := 1]
  data[PromoInterval == 'Jan,Apr,Jul,Oct'  & (month %in% c(1, 4, 7, 10)), newPromo2 := 1]
  data[PromoInterval == 'Mar,Jun,Sept,Dec' & (month %in% c(3, 6, 9, 12)), newPromo2 := 1]
  data[, Promo2Since := as.numeric(Promo2Since)] # for backward compatability at least
  
  # CompetitionOpen stuff per day
  data[, daysSinceCompetitionOpen := as.numeric(Date - CompetitionOpenSince)]
  data[, CompetitionDistance2 := ifelse(Date >= CompetitionOpenSince, CompetitionDistance, NA)]
  data[, CompetitionOpenSince := as.numeric(CompetitionOpenSince)] # for backward compatability at least
  
  if (config$debug.preprocess1) {
    # The rest of the processing is slow, so...
    cat('NOTE: using a small subset of the data to debug the preprocessing code!\n')
    data = data[Store %in% c(100 * c(1:20), 1081), ]
  }

  # Cast store to integer (too many categories)
  # NOTE: has to be done after all merges (that need Store to be categorical)
  # FIXME: It's probably best to just remove this, the idea is to include other 
  # features that will make this redundant
  data[, Store := as.integer(Store)]

  # (helper: create a new training example range, based on an existing one)
  new.examples = function(store, date.new1, date.new2, date.old) {
    idx = which(data$Store == store & data$Date == date.old)
    if (length(idx) == 0) return (NULL)
    dw = as.numeric(date.new2 - date.new1) + 1
    new.rows = data[rep(idx, dw), ]
    new.rows[, Date := date.new1 + (0:(dw - 1))]
    new.rows[, DayOfWeek := (wday(Date) - 2) %% 7 + 1]
    new.rows[, year   := as.integer(format(Date, '%y'))]
    new.rows[, month := as.integer(format(Date, '%m'))]
    new.rows[, day   := as.integer(format(Date, '%d'))]
    new.rows[, absday := as.integer(Date)]
    new.rows[, Id := -1e7]
    new.rows[, Sales := 0]
    new.rows[, Customers := 0]
    new.rows[, Open := 0]
    new.rows[, Promo := 0]
    # FIXME weather, holidays... (shouldn't be important)
    return (new.rows)
  }
  
  # Make sure sorted by dates per store at this point
  data = data[order(Store, Date)]

  # Deal with "refurbished" stores
  RefurbishedStores = as.numeric(subset(data.frame(table(train$Store)), Freq < 900)$Var1)
  data[, RefurbStore := 0]
  data[Store %in% RefurbishedStores, RefurbStore := 1]
  data[, AfterRefurb := -1]
  data[RefurbStore == 1 & Date < as.Date('2014-07-01'), AfterRefurb := 0]
  data[RefurbStore == 1 & Date > as.Date('2014-12-31'), AfterRefurb := 1]

  #
  cat(date(), 'Reintroducing missing dates\n')
  #
  
  # Reintroduce the missing dates, with as much info as possible.
  # This way all time series will then be regular.
  refurb.start.date = as.Date('2014-07-01')
  refurb.end.date = as.Date('2014-12-31')
  for (st in RefurbishedStores) {
    data = rbind(data, new.examples(st, refurb.start.date, refurb.end.date, refurb.start.date - 1))
  }
  
  # Fix store 988 that for some reason doens't have an entry for the first day, 2013-01-01
  data = rbind(data, new.examples(988, as.Date('2013-01-01'), as.Date('2013-01-01'), as.Date('2013-01-02')))

  # NOTE: At this point we should have all dates for all stores, howerver recall
  # that many sotres simply don't appear in the testset.

  # Add the time since (guessed) store opening/re-opening date
  old.date = as.Date('2000-01-01') # some arbitrary old date to signal the stores already in steady state throughout the train+test period
  data[, OpeningDate := old.date]
  data[Open == 1, OpeningDate := min(Date), by = Store]
  data[, OpeningDate := max(OpeningDate), by = Store] # assuming this happend at most once per store (could check...)
  data[OpeningDate < as.Date('2013-02-01'), OpeningDate := old.date]
  #data[RefurbStore == 1, OpeningDate := as.Date('2015-01-01'), by = Store] # not sure what to do with this one, since I don't believe all stores were actually closed for the entire period missing
  data[, DaysSinceOpening := 0]
  data[Date > OpeningDate, DaysSinceOpening := as.numeric(Date - OpeningDate)]
  data[, OpeningDate := NULL]

  #
  cat(date(), 'Merging external information\n')
  #

  if (1) {
    # Weather data for each state
    weather = NULL
    
    for (state in levels(store$State)) {
      ext = fread(paste0(config$datadir, '/EXT_weather_', state, '.csv'), stringsAsFactors = F)
      ext[, Date := as.Date(Date)]
      ext[, State := state]
      ext[, Event := 0]
      ext[Events %in% c('Fog', 'Fog-Rain', 'Rain', 'Rain-Hail'), Event := 1]
      ext[Events %in% c('Fog-Rain-Hail', 'Fog-Rain-Hail-Thunderstorm', 'Fog-Rain-Thunderstorm', 'Fog-Thunderstorm', 'Rain-Hail-Thunderstorm', 'Thunderstorm', 'Rain-Thunderstorm'), Event := 2]
      ext[Events %in% c('Fog-Rain-Snow', 'Fog-Rain-Snow-Hail', 'Fog-Snow', 'Fog-Snow-Hail', 'Rain-Snow', 'Rain-Snow-Hail', 'Snow', 'Snow-Hail'), Event := 3]
      ext[Events %in% c('Rain-Snow-Hail-Thunderstorm', 'Rain-Snow-Thunderstorm'), Event := 4]
      ext = ext[, .(Date, State, Max_TemperatureC, Min_TemperatureC, Precipitationmm, Event)]
      weather = rbind(weather, ext)
    }
    
    weather[, State := factor(State, levels = levels(store$State))]
    data = merge(data, weather, by = c('Date', 'State'), all.x = T)
    rm(weather)
  }
  
  if (1) {
    # Google trends data for "rossmann" in each state
    # TODO: this doesn't seem to be very interesting, but maybe I need to look
    # for higher resolution and more relevant similar signals.
    google = NULL
    
    for (state in levels(store$State)) {
      ext = fread(paste0(config$datadir, '/EXP_gtrends_DE_', state, '.csv'), stringsAsFactors = F) # should be EXT_... nevermind
      setnames(ext, 'rossmann', 'google')
      ext[, Date := as.Date(Week, '%F')]
      ext = ext[Date >= '2012-12-20' & Date <= '2015-12-20', .(Date, google)]
      ext[, google2 := lowess(ext$Date, ext$google, f = 0.05)$y]
      ext[, State := state]
      google = rbind(google, ext)
    }

    google[, State := factor(State, levels = levels(store$State))]
    data = merge(data, google, by = c('Date', 'State'), all.x = T)

    #ext = fread(paste0(config$datadir, '/EXP_gtrends_DE.csv'), stringsAsFactors = F)
    #setnames(ext, 'rossmann', 'google3')
    #ext[, Date := as.Date(Week, '%F')]
    #ext = ext[Date >= '2012-12-20' & Date <= '2015-12-20', .(Date, google3)]
    #ext[, google4 := lowess(ext$Date, ext$google, f = 0.05)$y]
    #data = merge(data, ext, by = 'Date', all.x = T)
    #rm(ext)
  }
  
  if (1) {
    # Consumer price index
    cpi = fread(paste0(config$datadir, '/EXT_CSI.csv'), stringsAsFactors = F) # why did I call it cSi !? nevermind...
    cpi[, Date := as.Date(paste(Year, Month, '01', sep = '-'))]
    cpi = cpi[, .(Date, CSIall, CSIfood)]
    
    data = merge(data, cpi, by = 'Date', all.x = T)
    data = data[order(Store, Date)]
    data[, CSIall  := na.locf(CSIall , na.rm = F), by = Store]
    data[, CSIfood := na.locf(CSIfood, na.rm = F), by = Store]
    data[, CSIall2  := lowess(Date, CSIall , f = 0.05)$y, by = Store]
    data[, CSIfood2 := lowess(Date, CSIfood, f = 0.05)$y, by = Store]
  }
  
  if (0) {
    # Additional macroeconomic indicators for Germany
    # TODO: well... this seems a little useless. I need to find a better source with
    # a higher resolution, and a information from the test period...
    
    for (ind in c('salesMoM', 'GfK_CCI', 'CPIMoM')) {
      ext = fread(paste0(config$datadir, '/EXP_', ind, '.csv'), stringsAsFactors = F) # should be EXT_... nevermind
      ext[, Date := as.Date(DateTime, '%m/%d/%Y %T')]
      ext[, DateTime := NULL]
      ext = ext[Date >= as.Date('2013-01-01') & Date <= as.Date('2015-09-17'), ]
      ext = ext[order(ext$Date), ]
    }
  }
  
  # TODO: additional external signals
  
  # Make sure at this point dates are sorted
  data = data[order(Store, Date)]
  
  # NOTE: Everything excpet Sales and Customers based signals is present in
  # the testset, so this is leakage we can exploit regardless of lag.
  
  # Rolling statistics: promo
  data[, rPromo1 := shift(Promo, 1, type = 'lag' , fill = 0.5), by = Store] # did store have promo yesterday
  data[, lPromo1 := shift(Promo, 1, type = 'lead', fill = 0.5), by = Store] # will store have promo tomorrow

  if (0) {
    data[, rPromo.sum3  := rollapply(Promo,  3, sum, align = 'right', fill = 1.5), by = Store]
    data[, rPromo.sum7  := rollapply(Promo,  7, sum, align = 'right', fill = 3.5), by = Store]
    data[, lPromo.sum3  := rollapply(Promo,  3, sum, align = 'left' , fill = 1.5), by = Store]
    data[, lPromo.sum7  := rollapply(Promo,  7, sum, align = 'left' , fill = 3.5), by = Store]
    data[, cPromo.sum3  := rollapply(Promo,  3, sum,                  fill = 1.5), by = Store]
    data[, cPromo.sum7  := rollapply(Promo,  7, sum,                  fill = 3.5), by = Store]
    data[, rPromo.sum30 := rollapply(Promo, 30, sum, align = 'right', fill =  15), by = Store]
    data[, rPromo.sum90 := rollapply(Promo, 90, sum, align = 'right', fill =  45), by = Store]
  }
  
  data[, rPromo.wd := shift(Promo, 1, type = 'lag' , fill = 0.5), by = list(Store, DayOfWeek)] # did store have promo last week
  data[, lPromo.wd := shift(Promo, 1, type = 'lead', fill = 0.5), by = list(Store, DayOfWeek)] # will store have promo next week
  
  # Rolling statistics: open
  data[, rOpen1 := shift(Open, 1, type = 'lag' , fill = 0.5), by = Store] # was store open yesterday
  data[, lOpen1 := shift(Open, 1, type = 'lead', fill = 0.5), by = Store] # will store be open tomorrow
  data[, rOpen.sum3  := rollapply(Open,  3, sum, align = 'right', fill = 1.5), by = Store]
  data[, rOpen.sum7  := rollapply(Open,  7, sum, align = 'right', fill = 3.5), by = Store]
  data[, lOpen.sum3  := rollapply(Open,  3, sum, align = 'left' , fill = 1.5), by = Store]
  data[, lOpen.sum7  := rollapply(Open,  7, sum, align = 'left' , fill = 3.5), by = Store]
  data[, cOpen.sum3  := rollapply(Open,  3, sum,                  fill = 1.5), by = Store]
  data[, cOpen.sum7  := rollapply(Open,  7, sum,                  fill = 3.5), by = Store]
  data[, rOpen.sum30 := rollapply(Open, 30, sum, align = 'right', fill =  15), by = Store]
  data[, lOpen.sum30 := rollapply(Open, 30, sum, align = 'left' , fill =  15), by = Store]
  data[, cOpen.sum30 := rollapply(Open, 30, sum,                  fill =  15), by = Store]
  
  # Rolling statistics: holiday
  data[, rHoliday1 := as.numeric(shift(StateHoliday, 1, type = 'lead', fill = 0.5)), by = Store] # was there a holiday yesterday
  data[, lHoliday1 := as.numeric(shift(StateHoliday, 1, type = 'lag' , fill = 0.5)), by = Store] # will there be a holiday tomorrow
  data[, lHoliday.any3 := as.numeric(rollapply(StateHoliday, 3, function(x) any(x != '0'), align = 'left', fill = 0.5)), by = Store]
  data[, lHoliday.any7 := as.numeric(rollapply(StateHoliday, 7, function(x) any(x != '0'), align = 'left', fill = 0.5)), by = Store]
  data[, cHoliday.any15 := as.numeric(rollapply(StateHoliday, 15, function(x) any(x != '0'), align = 'center', fill = 0.5)), by = Store]
  
  # Rolling statistics: weather
  data[, rEvent1 := as.numeric(shift(Event, 1, type = 'lead', fill = 0.5)), by = Store] # was there a weather event yesterday
  data[, lEvent1 := as.numeric(shift(Event, 1, type = 'lag' , fill = 0.5)), by = Store] # will there be a weather event tomorrow
  
  # TODO: I need better features for long closures
  if (0) {
    count.closed.days = function(x, do.rev) {
      n = nrow(x)
      y = rep(0, n)
      if (n >= 3) {
        xx = x$Open
        if (do.rev) xx = rev(xx)
        for (i in 2:n) {
          y[i] = ifelse(xx[i - 1], 0, y[i - 1] + 1)
        }
        if (do.rev) y = rev(y)
      }
      return (y)
    }
    
    # days since last close
    data[, daysClosedP := count.closed.days(.SD, F), by = Store]
    # days to next close (leak!)
    data[, daysClosedF := count.closed.days(.SD, T), by = Store]
  
    # TODO: how recently was the recent nontrivial closure and for how long
  }
  
  # Extend weekly google trends data to daily with ZOH
  data[, google  := na.locf(google , na.rm = F), by = Store]
  data[, google2 := na.locf(google2, na.rm = F), by = Store]
  
  # This is used as the response by some models so just add it
  # NOTE: but remember not to include it as a feature...
  data[, logSales := log1p(Sales)]
  
  return (list(data = data, valid.truth = valid.truth))
}

config$preprocess2 = function(config, pp1.data) {
  lag.days   = config$lag.days
  lag.weeks  = ceiling(lag.days /  7)
  lag.months = ceiling(lag.days / 31)

  # Take only those variable we need to create lagged features for
  # Also: make sure at this point dates are sorted
  data = pp1.data[order(Store, Date), .(Id, Store, StoreType, Assortment, State, Date, absday, day, DayOfWeek, Promo, Open, Sales, Customers)]

  cat(date(), 'PP2: Basic lag stuff\n')
  
  # I will remove these later, but need them to treat "0" sales correctly
  data[, naSales     := Sales]
  data[, naCustomers := Customers]
  data[naSales     == 0, naSales     := NA]
  data[naCustomers == 0, naCustomers := NA]

  # Lagged sales, customers, and sales per customer. Promo, day, week or month
  # based (FIXME do we need more/less combinations?)
  data[, sSales        := shift(naSales    , lag.days  ), by = Store                        ]
  data[, sCustomers    := shift(naCustomers, lag.days  ), by = Store                        ]
  data[, sSales.p      := shift(naSales    , lag.days  ), by = list(Store, Promo)           ]
  data[, sCustomers.p  := shift(naCustomers, lag.days  ), by = list(Store, Promo)           ]
  data[, sSales.w      := shift(naSales    , lag.weeks ), by = list(Store, DayOfWeek)       ]
  data[, sSales.wp     := shift(naSales    , lag.weeks ), by = list(Store, DayOfWeek, Promo)]
  data[, sCustomers.wp := shift(naCustomers, lag.weeks ), by = list(Store, DayOfWeek, Promo)]
  data[, sSales.mp     := shift(naSales    , lag.months), by = list(Store, day, Promo)      ]
  data[, sCustomers.mp := shift(naCustomers, lag.months), by = list(Store, day, Promo)      ]

  data[, sSales1 := shift(naSales, lag.days + 0), by = Store] # will not impute NAs in this one, so not the same as sSales
  data[, sSales2 := shift(naSales, lag.days + 1), by = Store]
  data[, sSales3 := shift(naSales, lag.days + 2), by = Store]
  data[, sSales4 := shift(naSales, lag.days + 3), by = Store]
  data[, sSales5 := shift(naSales, lag.days + 4), by = Store]
  data[, sSales6 := shift(naSales, lag.days + 5), by = Store]
  data[, sSales7 := shift(naSales, lag.days + 6), by = Store]
  
  # Define sales per customer (should be more comparable across stores)
  data[, sSPC    := sSales    / sCustomers   ]
  data[, sSPC.p  := sSales.p  / sCustomers.p ]
  data[, sSPC.wp := sSales.wp / sCustomers.wp]
  data[, sSPC.mp := sSales.mp / sCustomers.mp]

  # Mean sales per customer, across all stores / stores from same state
  data[, sSPC.mall := mean(sSPC, na.rm = T), by = Date                                    ]
  data[, sSPC.mta  := mean(sSPC, na.rm = T), by = list(Date, StoreType, Assortment       )]
 #data[, sSPC.mtas := mean(sSPC, na.rm = T), by = list(Date, StoreType, Assortment, State)]
  
  cat(date(), 'PP2: Lagged rolling statistics: customers by week\n')
  data[, week := isoweek(Date)]
  data[, year := isoyear(Date)]
  data[, sCustomersThisWeek := shift(cumsum(Customers), lag.days, fill = 0), by = .(Store, year, week)] # only nondegenerate for lag < 7
  weekly.customers = data[, sum(Customers), by = .(Store, year, week)]
  weekly.customers[, week := week + lag.weeks]
  setnames(weekly.customers, old = 'V1', new = 'sCustomersLastWeek')
  data = merge(data, weekly.customers, by = c('Store', 'year', 'week'), all.x = T)
  data = data[order(Store, Date)]
  data[, week := NULL]
  data[, year := NULL]
  
  cat(date(), 'PP2: Lagged rolling statistics: weekday+promo based sales\n')
  data[, sSales.wp.med4  := rollapply(sSales.wp,  4, median, align = 'right', na.rm = T, fill = NA), by = list(Store, DayOfWeek, Promo)]
  data[, sSales.wp.med12 := rollapply(sSales.wp, 12, median, align = 'right', na.rm = T, fill = NA), by = list(Store, DayOfWeek, Promo)]

  if (config$debug.preprocess2) {
    cat('NOTE: debugging => no fancy features\n')
  } else {
    cat(date(), 'PP2: Lagged rolling statistics: day-to-day sales\n')
    data[, sSales.mean7   := rollapply(sSales, 7, mean  , align = 'right', na.rm = T, fill = NA), by = Store]
    data[, sSales.median7 := rollapply(sSales, 7, median, align = 'right', na.rm = T, fill = NA), by = Store]
    data[, sSales.min7    := rollapply(sSales, 7, min   , align = 'right', na.rm = T, fill = NA), by = Store]
    data[, sSales.max7    := rollapply(sSales, 7, max   , align = 'right', na.rm = T, fill = NA), by = Store]
    cat(date(), 'PP2: Lagged rolling statistics: weekday+promo based customers\n')
    data[, sCustomers.wp.med4  := rollapply(sCustomers.wp,  4, median, align = 'right', na.rm = T, fill = NA), by = list(Store, DayOfWeek, Promo)]
   #data[, sCustomers.wp.med12 := rollapply(sCustomers.wp, 12, median, align = 'right', na.rm = T, fill = NA), by = list(Store, DayOfWeek, Promo)]
    
    cat(date(), 'PP2: Lagged rolling statistics: weekday+promo based sales per customer\n')
    data[, sSPC.wp.med4       := rollapply(sSPC.wp     ,  4, median, align = 'right', na.rm = T, fill = NA), by = list(Store, DayOfWeek, Promo)]
   #data[, sSPC.wp.med12      := rollapply(sSPC.wp     , 12, median, align = 'right', na.rm = T, fill = NA), by = list(Store, DayOfWeek, Promo)]

    data[, sSPC.mta.med7   := rollapply(sSPC.mta ,  7, median, align = 'right', na.rm = T, fill = NA), by = Store]
   #data[, sSPC.mtas.med4  := rollapply(sSPC.mtas,  4, median, align = 'right', na.rm = T, fill = NA), by = Store]
   #data[, sSPC.mtas.med12 := rollapply(sSPC.mtas, 12, median, align = 'right', na.rm = T, fill = NA), by = Store]

    if (0) { # (it's very slow and quite useless)
      # Very simplistic linear prediction
      # - ignores irregular sampling intervals due to grouping by Promo
      # - ignores Open
      # - ignores other important covariates
      linear.pred = function(z, horizon) {
        n = length(z)
        if (sum(is.na(z)) >= n - 2) return (0) # can't return NA (not supported by rollapply)
        x = 1:n
        beta = coef(lm(z ~ x))
        pred = sum(beta * c(1, n + horizon))
        pred = ifelse(pred > 0, pred, 1)
        return (pred)
      }
  
      cat(date(), 'PP2: Simple linear predictions from previous weekday+promo\n')
      data[, sSales.wp.lin4      := rollapply(sSales.wp,  4, linear.pred, horizon = lag.weeks, align = 'right', fill = NA), by = list(Store, DayOfWeek, Promo)]
      data[, sSales.wp.lin12     := rollapply(sSales.wp, 12, linear.pred, horizon = lag.weeks, align = 'right', fill = NA), by = list(Store, DayOfWeek, Promo)]
      data[sSales.wp.lin4  == 0, sSales.wp.lin4  := NA]
      data[sSales.wp.lin12 == 0, sSales.wp.lin12 := NA]
    
      data[, sCustomers.wp.lin4  := rollapply(sCustomers.wp,  4, linear.pred, horizon = lag.weeks, align = 'right', fill = NA), by = list(Store, DayOfWeek, Promo)]
      data[, sCustomers.wp.lin12 := rollapply(sCustomers.wp, 12, linear.pred, horizon = lag.weeks, align = 'right', fill = NA), by = list(Store, DayOfWeek, Promo)]
      data[, sSPC.wp.lin4        := rollapply(sSPC.wp      ,  4, linear.pred, horizon = lag.weeks, align = 'right', fill = NA), by = list(Store, DayOfWeek, Promo)]
      data[, sSPC.wp.lin12       := rollapply(sSPC.wp      , 12, linear.pred, horizon = lag.weeks, align = 'right', fill = NA), by = list(Store, DayOfWeek, Promo)]
      data[, sSPC.mtas.lin4      := rollapply(sSPC.mtas    ,  4, linear.pred, horizon = lag.weeks, align = 'right', fill = NA), by = Store]
      data[, sSPC.mtas.lin12     := rollapply(sSPC.mtas    , 12, linear.pred, horizon = lag.weeks, align = 'right', fill = NA), by = Store]
    }

    if (0) { # (it's absurdly slow, about two hours? it's also pretty crappy since it ignores so many huge effects like xmas...)
      # Rolling lagged linear model predictions
      rllmp = function(dt, w, h) {
        n = nrow(dt)
        wph = w + h
        preds = rep(NA, n)
        if (wph > n) return (preds)
        woffs = (1:w) - wph
        for (t in wph:n) {
          te = dt[t, ]
          if (te$Open == 0) {
            preds[t] = 0
          } else if (sum(!is.na(dt$Sales[t + woffs])) > 15) {
            tr = dt[t + woffs, ]
            preds[t] = predict(lm(Sales ~ absday + DayOfWeek + Open + Promo, tr), te)
          }
        }
        return (preds)
      }
      
      data[, sSales.lin := rllmp(.SD, w = 30, h = lag.days), by = Store]
    }
  }
  
  cat(date(), 'PP2: Cleanup\n')
  
  # Where this makes sense: Clean up missing values using the last available ones
  # TODO add many other similar signals that might benefit from it...
  # (I'm not sure if this helps or makes things worse...)
  data[, sSales        := na.locf(sSales       , na.rm = F), by = Store                        ] 
  data[, sCustomers    := na.locf(sCustomers   , na.rm = F), by = Store                        ] 
  data[, sSPC          := na.locf(sSPC         , na.rm = F), by = Store                        ] 
  data[, sSales.p      := na.locf(sSales.p     , na.rm = F), by = list(Store, Promo)           ] 
  data[, sCustomers.p  := na.locf(sCustomers.p , na.rm = F), by = list(Store, Promo)           ] 
  data[, sSPC.p        := na.locf(sSPC.p       , na.rm = F), by = list(Store, Promo)           ] 
  data[, sSales.wp     := na.locf(sSales.wp    , na.rm = F), by = list(Store, DayOfWeek, Promo)]
  data[, sCustomers.wp := na.locf(sCustomers.wp, na.rm = F), by = list(Store, DayOfWeek, Promo)]
  data[, sSPC.wp       := na.locf(sSPC.wp      , na.rm = F), by = list(Store, DayOfWeek, Promo)]
  data[, sSPC.mall     := na.locf(sSPC.mall    , na.rm = F), by = Store                        ]
  data[, sSPC.mta      := na.locf(sSPC.mta     , na.rm = F), by = Store                        ]
 #data[, sSPC.mtas     := na.locf(sSPC.mtas    , na.rm = F), by = Store                        ]
  data[, sSales.mp     := na.locf(sSales.mp    , na.rm = F), by = list(Store, day, Promo)      ]
  data[, sCustomers.mp := na.locf(sCustomers.mp, na.rm = F), by = list(Store, day, Promo)      ]
  data[, sSPC.mp       := na.locf(sSPC.mp      , na.rm = F), by = list(Store, day, Promo)      ]

  # Keep only Id and the new lagged features
  data[, Store       := NULL]
  data[, StoreType   := NULL]
  data[, Assortment  := NULL]
  data[, State       := NULL]
  data[, Date        := NULL]
  data[, day         := NULL]
  data[, absday      := NULL]
  data[, DayOfWeek   := NULL]
  data[, Promo       := NULL]
  data[, Open        := NULL]
  data[, Sales       := NULL]
  data[, Customers   := NULL]
  data[, naSales     := NULL]
  data[, naCustomers := NULL]

  id.idx = which(names(data) == 'Id')
  names(data) = paste0('lag.', substring(names(data), 2))
  names(data)[id.idx] = 'Id'
  
  return (data)
}

config$preprocess3 = function(config, data) {
  if (1) {
    # Experiment: PCA on dates
    # The idea: reduce the dimensionality of the features of all stores pertaining to a given day. 

    # FIXME: this only seems potentially useful if I can somehow use sales and customers too, but of
    # course, I can't use the sales in day d to predict for day d, I can only use the sales up to
    # day (d - lag).
    
    # FIXME this should be done in preprocess1/2 ?
    
    cat(date(), 'Adding PC scores based on dates\n')

    df = data[Store %in% unique(data[Id > 0, Store]), .(Store, Date, year ,month, day, intDayOfWeek = as.numeric(DayOfWeek), SchoolHoliday, Open, Promo)]
    df = as.data.frame(dcast(df, Date + year + month + day + intDayOfWeek ~ Store, value.var = c('Open', 'Promo', 'SchoolHoliday')))
    df = df[, c(T, apply(df[, 2:ncol(df)], 2, var) != 0)] # (remove constant columns so that we can normalize)
    prcomp.res = prcomp(df[, 2:ncol(df)], center = T, scale. = T)
    
    if (0) {
      plot(prcomp.res, type = 'l', log = 'y', npcs = 60)
      df2 = cbind(data[Store == '600', .(year, month, day, DayOfWeek)], prcomp.res$x[, 1:15])
      boxplot(PC6 ~ year, df2)
      boxplot(PC6 ~ month, df2)
      boxplot(PC6 ~ day, df2)
      boxplot(PC6 ~ DayOfWeek, df2)
      # (other expected strong effects are StateHoliday x State, the data balck hole in 2014, xmas, ....)
    }
    
    pcs = data.table(Date = df$Date, prcomp.res$x[, 1:15])
    rm(prcomp.res)
    names(pcs)[-1] = paste0('date', names(pcs)[-1])
    data = merge(data, pcs, by = 'Date')

    # Fix the ordering
    data = data[order(Store, Date)]
  }
  
  # Split the data back into train and test sets
  dsets = list()
  dsets$train = data[data$Id <  0, ]
  dsets$test  = data[data$Id >= 0, ]

  if (config$validation.scheme != 'none') {
    # Hold out part of the data for performance validation

    if (config$hide.validation) {
      # Split decided on earlier, and responses hidden away
      valid.idxs = dsets$train$Id %in% config$valid.truth$Id

      dsets$valid = dsets$train[ valid.idxs, ]
      dsets$train = dsets$train[!valid.idxs, ]
    
      # At this point it is (hopefully) safe to reintroduce the validation ground
      # truth. The idea is that processing proceeds separately for training and 
      # validation sets, and that validation targets will only be used for.. validation
      dsets$valid = dsets$valid[order(Store, Date), ]
      dsets$valid[, Sales := as.numeric(config$valid.truth[order(Store, Date), ]$Sales)]
      dsets$valid = dsets$valid[Sales != 0, ] # zero sales are ignored by the evaluation metric anyway so no point to carry them around
      dsets$valid[, logSales := log1p(Sales)]
    } else {
      # Split now
      valid.idxs = config$get.validation.idxs(config, dsets$train, dsets$test)
      dsets$valid = dsets$train[ valid.idxs, ]
      dsets$train = dsets$train[!valid.idxs, ]
      
      if (1) {
        cat('NOTE: Removing stores not in testset from the validset\n')
        dsets$valid = dsets$valid[Store %in% unique(dsets$test$Store)]
      }
    }
  }

  if (0) {
    # Generate a ranked store feature (which may be more useful in tree-based 
    # modeling than the original store id, though will lose any usable leakage
    # that might be in the latter...)
    
    # NOTE: this does some fitting on the trainset of course, and using this as
    # a feature in the main modeling to come is harmful leakage. But I think it
    # might be worth it (and I will hopefully see it in the validation error if 
    # not).
    
    ranks = dsets$train[, mean(Sales), by = 'Store']
    ranks[, rnkStoreSales := rank(V1)]
    ranks[, V1 := NULL]
    dsets$train = merge(dsets$train, ranks, by = 'Store')
    dsets$test = merge(dsets$test, ranks, by = 'Store')
    if (!is.null(dsets$valid)) {
      dsets$valid = merge(dsets$valid, ranks, by = 'Store')
    }
    
    ranks = dsets$train[, mean(Customers), by = 'Store']
    ranks[, rnkStoreCust := rank(V1)]
    ranks[, V1 := NULL]
    dsets$train = merge(dsets$train, ranks, by = 'Store')
    dsets$test = merge(dsets$test, ranks, by = 'Store')
    if (!is.null(dsets$valid)) {
      dsets$valid = merge(dsets$valid, ranks, by = 'Store')
    }
    
    ranks = dsets$train[, mean(Sales / Customers, na.rm = T), by = 'Store']
    ranks[, rnkStoreSPC := rank(V1)]
    ranks[, V1 := NULL]
    dsets$train = merge(dsets$train, ranks, by = 'Store')
    dsets$test = merge(dsets$test, ranks, by = 'Store')
    if (!is.null(dsets$valid)) {
      dsets$valid = merge(dsets$valid, ranks, by = 'Store')
    }
  }
  
  if (config$include.pcs > 0) {  
    # Use SVD to generate continuous store structure features
    
    # We already know about several strong components: StoreType, Assortment, 
    # CompetitionDistance, Promo2, PromoInterval, RefurbStore, sunStore, State.
    # One thing we could do is regress these out somehow or (in the case of a 
    # few categorical factors) do the SVD separately on each group and 
    # concatenate the result, in an order that reflects the mean Sales per 
    # group. But I guess I don't care about redundancy much, so I can just do
    # SVD on the whole matrix, and include more PCs (which will most/all include
    # a lot information already found in the known factors).
    
    # Clearly this too will look better in sample than it really is. I don't 
    # want to sacrifice data to do this separately, so I'm not sure what to do
    # about this really. Hope for the best I guess.

    if (0) {
      cat('NOTE: Adding PC scores based on logSales\n')
      df = as.data.frame(dcast(dsets$train[, .(Store, Date, logSales)], Store ~ Date, value.var = 'logSales'))
    } else if (0) {
      cat('NOTE: Adding PC scores based on logSales in all-stores-open trainset days\n')
      pca.dates = dsets$train[, Date[all(Open != '0')], by = Date]$V1
      df = dsets$train[Date %in% pca.dates]
      df = as.data.frame(dcast(df[, .(Store, Date, logSales)], Store ~ Date, value.var = 'logSales'))
    } else if (0) {
      cat('NOTE: Adding PC scores based on logSales residuals after regressing out store features\n')
      df = dsets$train[, .(Store, Date, logSales, Open, StoreType, Assortment, imputedCompetitionDistance, Promo2, PromoInterval)]
      df[, logSales.resid := resid(lm(logSales ~ Open + StoreType + Assortment + imputedCompetitionDistance + Promo2 + PromoInterval, df))]
      df = as.data.frame(dcast(df[, .(Store, Date, logSales.resid)], Store ~ Date, value.var = 'logSales.resid'))
    } else {
      cat('NOTE: Adding PC scores based on everything we know about stores\n')
      df = dsets$train[, .(Store, Date, Sales, Customers, Promo, intStoreType, intAssortment, imputedCompetitionDistance, Promo2, intPromoInterval = as.numeric(PromoInterval), imputedCompetitionOpenSince, imputedPromo2Since, intState = as.numeric(State))]
      df[, medSales     := median(Sales    ), by = Store]
      df[, medCustomers := median(Customers), by = Store]
      df[Sales     == 0, Sales     := medSales    ]
      df[Customers == 0, Customers := medCustomers]
      df[, Sales     := log(Sales    )]
      df[, Customers := log(Customers)]
      df = as.data.frame(dcast(df, Store + intStoreType + intAssortment + imputedCompetitionDistance + Promo2 + intPromoInterval + imputedCompetitionOpenSince + imputedPromo2Since + intState ~ Date, value.var = c('Sales', 'Customers', 'Promo')))
    }
    
    df = df[, apply(df, 2, var, na.rm = T) != 0] # (remove constant columns so that we can normalize)
    prcomp.res = prcomp(df[, 2:ncol(df)], center = T, scale. = T)
    if (0) {
      # Examine the PCs and their relation to known factors
      plot(prcomp.res, npcs = 60, type = 'l', log = 'y')
      df2 = cbind(dsets$train[Date == '2015-05-05', .(StoreType, RefurbStore, sunStore, Assortment, State)], prcomp.res$x[, 1:10])
      boxplot(PC1 ~ StoreType, df2)
      boxplot(PC1 ~ Assortment, df2)
      boxplot(PC1 ~ RefurbStore, df2)
      boxplot(PC1 ~ sunStore, df2)
      boxplot(PC1 ~ State, df2)
      boxplot(PC1 ~ StoreType : RefurbStore : sunStore : Assortment, df2)
    }
    pcs = data.table(Store = df$Store, prcomp.res$x[, 1:30])
    names(pcs)[-1] = paste0('store', names(pcs)[-1])
    rm(prcomp.res)
    
    dsets$train = merge(dsets$train, pcs, by = 'Store')
    dsets$test  = merge(dsets$test , pcs, by = 'Store')
    if (!is.null(dsets$valid)) {
      dsets$valid = merge(dsets$valid, pcs, by = 'Store')
    }

    if (0) {
      # Experiment: add PCs also based on Customers and SPC
      cat('NOTE: Adding PC scores based on customers and SPC\n')
      
      df = dsets$train[, .(Store, Date, Sales, Customers, Open, StoreType, Assortment, imputedCompetitionDistance, Promo2, PromoInterval, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10)]
      df[, z := resid(lm(log1p(Customers) ~ Open + StoreType + Assortment + imputedCompetitionDistance + Promo2 + PromoInterval + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10, df))]
      df = as.data.frame(dcast(df[, .(Store, Date, z)], Store ~ Date, value.var = 'z'))
      prcomp.res = prcomp(df[, 2:ncol(df)])
      pcs = data.table(Store = df$Store, prcomp.res$x[, 1:(config$include.pcs)])
      names(pcs)[-1] = paste0('cust', names(pcs)[-1])
      dsets$train = merge(dsets$train, pcs, by = 'Store')
      dsets$test  = merge(dsets$test , pcs, by = 'Store')
      if (!is.null(dsets$valid)) {
        dsets$valid = merge(dsets$valid, pcs, by = 'Store')
      }
      
      df = dsets$train[, .(Store, Date, Sales, Customers, Open, StoreType, Assortment, imputedCompetitionDistance, Promo2, PromoInterval, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10, custPC1, custPC2, custPC3, custPC4, custPC5)]
      df[, z := resid(lm(I(log1p(Customers) - log1p(Sales)) ~ Open + StoreType + Assortment + imputedCompetitionDistance + Promo2 + PromoInterval + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + custPC1 + custPC2 + custPC3 + custPC4 + custPC5, df))]
      df = as.data.frame(dcast(df[, .(Store, Date, z)], Store ~ Date, value.var = 'z'))
      prcomp.res = prcomp(df[, 2:ncol(df)])
      pcs = data.table(Store = df$Store, prcomp.res$x[, 1:(config$include.pcs)])
      names(pcs)[-1] = paste0('spc', names(pcs)[-1])
      dsets$train = merge(dsets$train, pcs, by = 'Store')
      dsets$test  = merge(dsets$test , pcs, by = 'Store')
      if (!is.null(dsets$valid)) {
        dsets$valid = merge(dsets$valid, pcs, by = 'Store')
      }
    }
    
    if (0) {
      # Experiment: add PCs specific to StoreType A
      cat('NOTE: Adding PC scores based on logSales in stores of type "a"\n')
      df = dsets$train[StoreType == 'a', .(Store, Date, logSales, Open, Assortment, imputedCompetitionDistance, Promo2, PromoInterval, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10, custPC1, custPC2, custPC3, custPC4, custPC5, spcPC1, spcPC2, spcPC3, spcPC4, spcPC5)]
      df[, z := resid(lm(logSales ~ Open + Assortment + imputedCompetitionDistance + Promo2 + PromoInterval + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + custPC1 + custPC2 + custPC3 + custPC4 + custPC5 + spcPC1 + spcPC2 + spcPC3 + spcPC4 + spcPC5, df))]
      df = as.data.frame(dcast(df[, .(Store, Date, z)], Store ~ Date, value.var = 'z'))
      prcomp.res = prcomp(df[, 2:ncol(df)])
      pcs = data.table(Store = df$Store, prcomp.res$x[, 1:(config$include.pcs)])
      names(pcs)[-1] = paste0('sta', names(pcs)[-1])
      dsets$train = merge(dsets$train, pcs, by = 'Store', all.x = T)
      dsets$test  = merge(dsets$test , pcs, by = 'Store', all.x = T)
      if (!is.null(dsets$valid)) {
        dsets$valid = merge(dsets$valid, pcs, by = 'Store', all.x = T)
      }
    }

  if (0) {
    # Experiment: Add store PCs separately for each year (even though each year has a different amount of data available)
      # FIXME need to regress previous PCs, otherwise it isn't interesting
    for (yr in 13:15) {
      cat('NOTE: Adding PC scores based on sales in', yr, '\n')
      pca.dates = dsets$train[year == yr, Date[mean(Open != '0') > 0.5], by = Date]$V1
      df = dsets$train[Date %in% pca.dates]
      df = as.data.frame(dcast(df[, .(Store, Date, logSales)], Store ~ Date, value.var = 'logSales'))
      prcomp.res = prcomp(df[, 2:ncol(df)])
      pcs = data.table(Store = df$Store, prcomp.res$x[, 1:3])
      
      names(pcs)[-1] = paste0('y', yr, names(pcs)[-1])
      dsets$train = merge(dsets$train, pcs, by = 'Store')
      dsets$test  = merge(dsets$test , pcs, by = 'Store')
      if (!is.null(dsets$valid)) {
        dsets$valid = merge(dsets$valid, pcs, by = 'Store')
      }
    }
  }
  }
  
  if (config$bagging.mode) {
    # Bootstrap the training data.
    # NOTE: we could use the out of bag for more efficient validation (though we probably don't
    # want early stopping, just for measuring performance), but it's unclear how to combine the
    # out of bag errors from different repetitions.
    # NOTE: I'm doing this here to allow the PCA to use all the training data regardless of 
    # bootstrapping.
    dsets$train = dsets$train[sample(nrow(dsets$train), replace = T), ]
  }
  
  if (config$recode.nas) {
    cat('NOTE: Representing NAs with', config$recode.na.as, '\n')
    dsets$train[is.na(dsets$train)] = config$recode.na.as
    dsets$test [is.na(dsets$test )] = config$recode.na.as
    if (!is.null(dsets$valid)) {
      dsets$valid[is.na(dsets$valid)] = config$recode.na.as
    }
  }
  
  if (config$remove.holidays) {
    cat('NOTE: Removing holidays from trainset\n')
    weeks = strftime(dsets$train$Date, format = '%Y-%W')
    holiday.weeks = unique(strftime(unique(dsets$train[StateHoliday != '0', Date]), format = '%Y-%W'))
    dsets$train = dsets$train[!(weeks %in% holiday.weeks), ]
  }
  
  if (config$remove.zero.sales) {
    cat('NOTE: Dropping all trainset days with no sales or closed\n')
    # FIXME Not sure if keeping this in will help or damage
    dsets$train = dsets$train[(Sales > 0) & Open, ]
  }
  
  if (config$remove.outliers) {
    # Remove (manually annotated) outlying (store, date range) pairs from the training data.
    # NOTE: this will clearly not work with timeseries models...
    cat('NOTE: Removing manually annotated outliers from trainset\n')
    
    config$outliers = read.csv(paste0(config$datadir, '/outliers.csv'), colClasses = c('integer', 'Date', 'Date'))
    
    bad.idxs = rep(F, nrow(dsets$train))
    for (i in 1:nrow(config$outliers)) {
      bad.idxs = bad.idxs | (dsets$train$Store == config$outliers$store[i] & dsets$train$Date >= config$outliers$date1[i] & dsets$train$Date <= config$outliers$date2[i])
    }
    if (sum(bad.idxs) > 0) {
      dsets$train = dsets$train[!bad.idxs, ]
    }
    rm(bad.idxs)
  }
  
  if (config$remove.xmas) {
    cat('NOTE: Removing xmas period from trainset (aka "war on xmas")\n')
    dsets$train = dsets$train[!((Date < '2013-01-15') | (Date > '2013-11-20' & Date < '2014-01-15') | (Date > '2014-11-20' & Date < '2015-01-15')), ]
  }

  if (config$calibration.scheme != 'none') {
    # Regardless of whether or not this part is held out, keep track of the last month or so for
    # calibration (but it's very problematic to calibrate on training data...)
    
    dsets$calib = dsets$train[Date > '2015-06-13']
    if (!is.null(dsets$valid)) {
      dsets$calib = rbind(dsets$calib, dsets$valid[Open == 1 & Date > '2015-06-13'])
    }
    
    # Remove stores that don't appear in the testset, holidays
    dsets$calib = dsets$calib[Store %in% unique(dsets$test$Store)]
    dsets$calib = dsets$calib[StateHoliday == '0']
    
    # Get rid of many outliers in this period, if we haven't already
    if (!config$remove.outliers) {
      outliers = read.csv(paste0(config$datadir, '/outliers.csv'), colClasses = c('integer', 'Date', 'Date'))
      bad.idxs = rep(F, nrow(dsets$calib))
      for (i in 1:nrow(outliers)) bad.idxs = bad.idxs | (dsets$calib$Store == outliers$store[i] & dsets$calib$Date >= outliers$date1[i] & dsets$calib$Date <= outliers$date2[i])
      if (sum(bad.idxs) > 0) dsets$calib = dsets$calib[!bad.idxs, ]
      rm(bad.idxs)
    }
    
    # TODO: match DayOfWeek, Promo per store
  }
  
  return (dsets)
}

config$init.ts = function(config, dsets) {
  return (config)
}

config$shutdown.ts = function(config) {
  return (config)
}

config$train.ts = function(config, trainset, validset) {
  # Use all stores jointly to reduce noise (using a low rank representation)
  # FIXME do this separately per state?
  # FIXME the SVD makes some sales negative... maybe tuning the nr.pcs will resolve this
  # FIXME what do we do with the zero sales on closed days? clearly not MVN like this
  # FIXME do we need to scale and center?

  if (1) {
    # No fancy SVD
    return (trainset)
  }

  svd.filter = function(sd, pc.frac, min.pcs) {
    sales.matrix = as.data.frame(dcast(sd[, .(Store, Date, logSales)], Store ~ Date, value.var = 'logSales'))
    nr.pcs = ceiling(pc.frac * nrow(sales.matrix)) # FIXME tune this with CV..
    if (nr.pcs < min.pcs) return (sd$logSales)
    nr.dates = ncol(sales.matrix) - 1
    svd.res = svd(sales.matrix[, 1 + (1:nr.dates)], nu = nr.pcs, nv = nr.pcs)
    sales.matrix[, 1 + (1:nr.dates)] = svd.res$u %*% diag(svd.res$d[1:nr.pcs]) %*% t(svd.res$v)
    sales.matrix = as.data.table(sales.matrix)
    sales.matrix = melt(sales.matrix, id.vars = 1, measure.vars = 1 + (1:nr.dates), variable.name = 'Date', value.name = 'logSales')
    sales.matrix = sales.matrix[order(Store, Date)] # NOTE: assuming this is the order in trainset
    sales.matrix$logSales[sd$logSales == 0] = 0
    return (sales.matrix$logSales)
  }
  
  trainset[, logSalesClean := svd.filter(.SD, 0.8, 5), by = list(StoreType, Assortment, RefurbStore)]
  
  zidx = trainset$logSales == 0
  plot(trainset$logSales[!zidx], trainset$logSalesClean[!zidx], pch = '.', xlab = 'Raw log sales', ylab = 'Filtered log sales', main = 'Grouped low rank filtering')
  abline(0, 1, col = 2)

  clean.trainset = trainset
  clean.trainset[, logSales := logSalesClean]
  clean.trainset[, logSalesClean := NULL]
  
  # Everything else will be done in predict
  return (trainset)
}

config$predict.ts = function(config, model, testset) {
  # For now: simple model that applies separately to each series
  trainset = model
  validset = testset
  
  stores = unique(trainset$Store)
  nr.stores = length(stores)
  
  ts.model.select = 'tslm'
  
  preds.all = rep(NA, nrow(validset))  # NOTE: we actually do all prediction in this function and store them here
  
  for (i in 1:nr.stores) {
    if (i %% 100 == 0) cat(date(), 'Working on store', i, 'out of', nr.stores, '\n')
    
    # Extract data for this store
    train.df = as.data.frame(trainset[Store == stores[i], ])
    valid.df = as.data.frame(validset[Store == stores[i], ])

    # TODO (maybe during preprocessing):
    # - handle better: refurb stores, opening mid period, other long closures
    # - model buying spree before xmas, other holidays
    # - somehow downweight older dates in order to target testset
    
    if (1) {
      # If periodicity is not important for the model
      # (it may not be, since we know the dates and their translations into seasonsal signals)
      
      # Remove training days near long closures. These are often full of outliers.
      bad.idxs = which(train.df$cOpen.sum30 < 15)
      if (length(bad.idxs) > 0) {
        train.df = train.df[-bad.idxs, ]
      }
      
      # Remove all closed days (this seems to make things worse)
      #train.df = train.df[train.df$Open == 1, ]
      
      # Have to do this for validation too since for some stores we are left with fewer factor levels for some covars
      #valid.idxs = valid.df$Open == 1
      #valid.df = valid.df[valid.idxs, ]
      
      period = 7 # (this doesn't have an actual effect on the model, just allows me to keep some code below shared)
    } else {
      # If periodicity is important for the model

      # If always closed on Sun, remove it and predict 0 there, then have period = 6
      # FIXME: does this help at all? I think all models got this covered without
      # losing anything.
      sunday.idxs = which(train.df$DayOfWeek == 7)
      if (all(!train.df$Open[sunday.idxs])) {
        train.df = train.df[-sunday.idxs, ]
        raw.valid.df = valid.df
        valid.df = valid.df[-which(valid.df$DayOfWeek == 7), ]
        period = 6
      } else {
        period = 7
      }
      
      #valid.idxs = rep(T, nrow(valid.df))
    }
    
    if (config$take.log) {
      train.df$Sales = train.df$logSales
      valid.df$Sales = valid.df$logSales
    }

    regressors = c('absday', 'DayOfWeek', 'Promo', 'xmas', 'winter', 'spring', 'summer', 'cOpen.sum3', 'Event', 'lHoliday.any7')
    if (length(unique(train.df$Open)) != 1) regressors = c(regressors, 'Open')
    if (length(unique(train.df$inPromo2)) != 1) regressors = c(regressors, 'inPromo2')
    if (length(unique(train.df$AfterRefurb)) != 1) regressors = c(regressors, 'AfterRefurb')
    
    pred.h = nrow(valid.df) # prediction horizon
    pred.x = (nrow(train.df) + (1:pred.h) - 1) / period + 1
    pred.xlim = c(0.9 * nrow(train.df), nrow(train.df) + pred.h) / period
    #plot(train.df$Date[train.df$Sales != 0], train.df$Sales[train.df$Sales != 0], type = 'l')
    
    # Construct ts object
    train.ts = train.df; train.ts$Date = ts(train.ts$Date, frequency = period)
    valid.ts = valid.df; valid.ts$Date = ts(valid.ts$Date, frequency = period)
    #train.ts = data.frame(lapply(train.df, ts, frequency = period))
    #valid.ts = data.frame(lapply(valid.df, ts, frequency = period))
    
    # Train models for the Sales variable:
    
    if (ts.model.select == 'snaive') {
      # Using ARIMA(0, 0, 0)(0, 1, 0)m random walk with seasonality (i.e., repeat last week?)
      fit = snaive(train.ts$Sales, h = pred.h)
    } else if (ts.model.select == 'tslm') {
      # Using "seasonal" linear regression
      fit.lm = tslm(paste0('Sales ~ ', paste0(regressors, collapse = ' + ')), train.ts) # casts factors into numeric?
      #summary(fit.lm)
      fit = forecast(fit.lm, newdata = valid.ts, h = pred.h)
    } else if (ts.model.select == 'stlf') {
      # Using "stlf" based on arima (etf can't use regressors)
      fit = stlf(train.ts$Sales, h = pred.h, s.window = 'periodic', method = 'arima', ic = 'bic', xreg = train.df[, regressors], newxreg = valid.df[, regressors])
    } else if (ts.model.select == 'tbats') {
      # Using TBATS
      fit = tbats(train.df$Sales, seasonal.periods = c(period, 365.25 * period / 7), use.parallel = T) # NOTE paralellization
      fit = forecast(fit, h = pred.h)
    }

    preds = as.numeric(fit$mean)
    
    # Look at the predictions
    nzidx = valid.df$Sales != 0
    if (config$take.log) {
      store.rmspe = sqrt(mean((expm1(preds[nzidx]) / expm1(valid.df$Sales[nzidx]) - 1) ^ 2))
    } else {
      store.rmspe = sqrt(mean((preds[nzidx] / valid.df$Sales[nzidx] - 1) ^ 2))
    }
    #cat('RMSPE =', store.rmspe, '\n')
    if (!is.finite(store.rmspe)) browser()
    
    if (store.rmspe > 0.4) {
      plot(fit, xlim = pred.xlim, main = paste('Store', stores[i]))
      abline(v = (which(!train.df$Open) - 1) / period + 1, lty = 2)
      lines(pred.x, valid.df$Sales, col = 2)
    }

    # Reintroduce dropped days
    if (period == 6) {
      preds.e = rep(0, nrow(raw.valid.df))
      preds.e[raw.valid.df$DayOfWeek != 7] = preds
      preds = preds.e
    } else {
      #preds.e = rep(0, length(valid.idxs))
      #preds.e[valid.idxs] = preds
      #preds = preds.e
    }

    if (any(is.na(preds))) browser()
        
    preds.all[validset$Store == stores[i]] = preds # NOTE: assumes that the validset is ordered by date per store
  }
  
  return (preds.all)
}

config$tune.ts = function(config, trainset) {
}

config$predict.storewise = function(config, model, testset) {
  trainset = model
  validset = testset
  
  stores = unique(trainset$Store)
  nr.stores = length(stores)
  
  model.select = 'xgb' # { lm, glmnet, xgb }
  
  preds.all = rep(NA, nrow(validset))  # NOTE: we actually do all prediction in this function and store them here
  
  for (i in 1:nr.stores) {
    if (i %% 100 == 0) cat(date(), 'Working on store', i, 'out of', nr.stores, '\n')
    
    # Extract data for this store
    train.df = trainset[Store == stores[i], ]
    valid.df = validset[Store == stores[i], ]
    
    if (all(valid.df$Sales == 0)) {
      # Can happen under some validation schemes
      next
    }
    
    if (1) {
      # Remove training days near long closures. These are often full of outliers.
      bad.idxs = which(train.df$cOpen.sum30 < 15)
      if (length(bad.idxs) > 0) {
        train.df = train.df[-bad.idxs, ]
      }
    }
    
    regressors = c('absday', 'DayOfWeek', 'Promo', 'xmas', 'winter', 'spring', 'summer', 'cOpen.sum3', 'Event', 'lHoliday.any7')
    if (length(unique(train.df$Open)) != 1) regressors = c(regressors, 'Open')
    if (length(unique(train.df$inPromo2)) != 1) regressors = c(regressors, 'inPromo2')
    if (length(unique(train.df$AfterRefurb)) != 1) regressors = c(regressors, 'AfterRefurb')
    if (config$take.log) {
      frmla.major = as.formula(paste0('logSales ~ ', paste0(regressors, collapse = ' + ')))
      frmla.all = logSales ~ . - Id - Date - Sales - logSales - Customers - 1
      frmla.right = ~ . - Id - Date - Sales - logSales - Customers - 1
    } else {
      frmla.major = as.formula(paste0('Sales ~ ', paste0(regressors, collapse = ' + ')))
      frmla.all = Sales ~ . - Id - Date - Sales - logSales - Customers - 1
      frmla.right = ~ . - Id - Date - Sales - logSales - Customers - 1
    }
    
    if (model.select == 'lm') {
      fit.lm = lm(frmla.major, train.df)
      #summary(fit.lm)
      preds = rep(0, nrow(valid.df))
      preds[valid.df$Open == 1] = predict(fit.lm, newdata = valid.df[valid.df$Open == 1, ])
    } else if (model.select == 'glmnet') {
      #plot(cv.glmnet(as.formula(paste0('Sales ~ ', paste0(regressors, collapse = ' + '))), train.df))
      glmnet.fit = glmnet(frmla.major, train.df, lambda = exp(-5))
      preds = predict(glmnet.fit, valid.df)
    } else if (model.select == 'xgb') {
      if (config$take.log) {
        xgb.fobj = 'reg:linear'
        
        xgb.feval = function(preds, dtrain) {
          labels = getinfo(dtrain, 'label')
          labels = exp(labels)
          preds  = exp(preds )
          value = sqrt(mean(((labels - preds) / labels) ^ 2))
          return (list(metric = 'rmspe', value = value))
        }
        
        trainset.y = train.df$logSales
        validset.y = valid.df$logSales
      } else {
        train.w = 1e6 / train.df$Sales ^ 2
        
        xgb.fobj = function(preds, dtrain) {
          labels = getinfo(dtrain, 'label')
          return (list(grad = train.w * (preds - labels), hess = train.w))
        }
        
        xgb.feval = function(preds, dtrain) {
          labels = getinfo(dtrain, 'label')
          value = sqrt(mean(((labels - preds) / labels) ^ 2))
          return (list(metric = 'rmspe', value = value))
        }
        
        trainset.y = train.df$Sales
        validset.y = valid.df$Sales
      }
      
      xgb.pars = list(
        booster = 'gbtree', objective = 'reg:linear', eval_metric = xgb.feval,
        eta               = 0.1 , # shrinkage along boosting rounds (lower will slow training convergence)
       #gamma             = 0   , # single tree constraint (higher will tend to create less complex trees)
        max_depth         = 3   , # single tree constraint (effectively the maximum variable interactions per tree)
       #min_child_weight  = 0   , # single tree constraint (higher will tend to create less complex trees)
        subsample         = 0.8 , # bagging-like randomization per round
        colsample_bytree  = 0.7   # random-forest like randomization per round
       #num_parallel_tree = 1     # random forest size at every stage of boosting
      )

      xgb.trainset = xgb.DMatrix(model.matrix(frmla.right, data = train.df), missing = config$recode.na.as, label = trainset.y)
      xgb.validset = xgb.DMatrix(model.matrix(frmla.right, data = valid.df), missing = config$recode.na.as, label = validset.y)
      watchlist = list(train = xgb.trainset, valid = xgb.validset)
      
      x.mod.t = xgb.train(params = xgb.pars, data = xgb.trainset, nrounds = 100, verbose = 0)
      #x.mod.t = xgb.train(params = xgb.pars, data = xgb.trainset, nrounds = 100, watchlist = watchlist)
      #impo = xgb.importance(colnames(model.matrix(frmla, data = train.df)), model = x.mod.t)
      #print(impo[1:50, ])
      preds = predict(x.mod.t, xgb.validset)
    } else {
      stop('Unexpected model.select')
    }

    if (!any(is.finite(preds))) browser()
    
    # Look at the predictions
    if (config$take.log) {
      store.rmspe = sqrt(mean((expm1(preds) / expm1(valid.df$logSales) - 1) ^ 2, na.rm = T))
    } else {
      store.rmspe = sqrt(mean((preds / valid.df$Sales - 1) ^ 2, na.rm = T))
    }
    #cat('RMSPE =', store.rmspe, '\n')
    
    if (store.rmspe > 0.4) {
      plot(train.df$Date[train.df$Open == 1], train.df$Sales[train.df$Open == 1], type = 'l', xlim = as.Date(c('2015-04-15', '2015-07-30')), lwd = 2, xlab = 'Date', ylab = 'Sales', main = paste('Store', stores[i]))
      lines(valid.df$Date[valid.df$Open == 1], valid.df$Sales[valid.df$Open == 1], col = 2, lwd = 2)
      lines(valid.df$Date[valid.df$Open == 1], preds[valid.df$Open == 1], col = 4, lwd = 2, lty = 2)
    }

    preds.all[validset$Store == stores[i]] = preds # NOTE: assumes that the validset is ordered by date per store
  }
  
  return (preds.all)
}

config$init.xgb = function(config, dsets) {
  if (config$take.log) {
    config$xgb.fobj = 'reg:linear'
    
    config$xgb.feval = function(preds, dtrain) {
      labels = getinfo(dtrain, 'label')
      labels = expm1(labels) # FIXME can compute once
      preds  = expm1(preds )
      errs = ((labels - preds) / labels) ^ 2
      value = sqrt(mean(errs[labels != 0]))
      return (list(metric = 'rmspe', value = value))
    }
  } else {
    cat('NOTE: Using weighting scheme', config$weights.sel, '\n')
    
    if (config$weights.sel == 'uniform') {
      # This leads to the RMSPE on all the data, no matter how old etc.
      config$train.w = rep(1, nrow(dsets$train))
    } else if (config$weights.sel == 'step') {
      # There seems to be a point before which the sales themselves (don't know 
      #if also the P(sales|features)...) behave markedly differently.
      config$train.w = ifelse(dsets$train$Date < as.Date('2013-10-1'), 0.1, 1)
    } else if (config$weights.sel == 'linear') {
      config$train.w = 1 + as.numeric(dsets$train$Date - min(dsets$train$Date)) / as.numeric(max(dsets$train$Date) - min(dsets$train$Date))
    } else if (config$weights.sel == 'exponential') {
      config$train.w = 1 - exp(-2 * as.numeric(dsets$train$Date - min(dsets$train$Date)) / as.numeric(max(dsets$train$Date) - min(dsets$train$Date)))
    } else if (config$weights.sel == 'special') {
      # * Reduce importance of the time of year right before Christmas (say, Dec)
      # * Make the time of year covered by the test set more impotant than the rest
      # * Lower the importance of stores not in the testset
      config$train.w = rep(1, nrow(dsets$train))
      config$train.w[dsets$train$month == 12 | dsets$train$cOpen.sum30 < 20] = 0.5
      config$train.w[dsets$train$month %in% c(8, 9)] = 2
      idxs = dsets$train$Store %in% unique(dsets$test$Store)
      config$train.w[idxs] = 2 * config$train.w[idxs]
      config$train.w = config$train.w * (1 + as.numeric(dsets$train$Date - min(dsets$train$Date)) / as.numeric(max(dsets$train$Date) - min(dsets$train$Date)))
    } else {
      stop('Unexpected weights.sel')
    }
    
    # NOTE:
    # The 1e6 scaling factor below is to make the learning rate similar to that used for
    # log-transformed responses. Exposes again the numerical difficulties with
    # non-transformed values in this competition. But this is probably not a
    # catastrophy, since the errors will probably be proportional to the true
    # responses, so the weighted errors will tend to be on a similar scale.
    # I'm cutting off the sales at a minimum so that a few outliers won't get too 
    # much weight, assuming similar ones don't come up in the testset.
    
    config$train.w = 1e6 / pmax(dsets$train$Sales, 1500) ^ 2 * config$train.w * (length(config$train.w) / sum(config$train.w))
    config$train.w[dsets$train$Sales == 0] = 0
    
    if (0) {
      # Visualize the weights
      plot(sort(dsets$train$Date), config$train.w[order(dsets$train$Date)] / max(config$train.w), pch = '.', main = 'MSE training weights (normalized to max)', xlab = 'Date', ylab = 'Weight')
    }

    config$xgb.fobj = function(preds, dtrain) {
      labels = getinfo(dtrain, 'label')
      return (list(grad = config$train.w * (preds - labels), hess = config$train.w))
    }
    
    config$xgb.feval = function(preds, dtrain) {
      labels = getinfo(dtrain, 'label')
      errs = ((labels - preds) / labels) ^ 2
      value = sqrt(mean(errs[labels != 0]))
      return (list(metric = 'rmspe', value = value))
    }
  }
  
  config$xgb.fixed.pars = list(
    booster          = 'gbtree',
    objective        = config$xgb.fobj,
    eval_metric      = config$xgb.feval,
    verbose          = 1
  )

  config$xgb.tuned.pars = list(
    eta               = 0.02, # shrinkage along boosting rounds (lower will slow training convergence)
   #gamma             = 0   , # single tree constraint (higher will tend to create less complex trees)
    max_depth         = 10  , # single tree constraint (effectively the maximum variable interactions per tree)
   #min_child_weight  = 0   , # single tree constraint (higher will tend to create less complex trees)
    subsample         = 0.9 , # bagging-like randomization per round
    colsample_bytree  = 0.7   # random-forest like randomization per round
   #num_parallel_tree = 10    # random forest size at every stage of boosting
  )
  
  config$xgb.n.rounds = 10000
  config$xgb.early.stop.round = 200
  config$xgb.print.every.n = 100

  # Things that actually work:
  #config$xgb.dat.forumla = ~ year + month + day + as.numeric(DayOfWeek) + Promo + SchoolHoliday + Store + as.numeric(StoreType) + as.numeric(Assortment) + CompetitionDistance + Promo2 + as.numeric(PromoInterval) + as.numeric(CompetitionOpenSince) + as.numeric(Promo2Since) - 1
  #config$xgb.dat.forumla = ~ year + month + day + as.numeric(DayOfWeek) + Promo + SchoolHoliday + Store + as.numeric(StoreType) + as.numeric(Assortment) + CompetitionDistance + Promo2 + as.numeric(PromoInterval) + as.numeric(CompetitionOpenSince) + as.numeric(Promo2Since) - 1 + rnkStoreSales + CompetitionDistance2 + rnkStoreCust + rnkStoreSPC + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + google + Event + cOpen.sum7 + lHoliday.any7 + Max_TemperatureC + Min_TemperatureC + Precipitationmm + StateHoliday + AfterRefurb + DaysSinceOpening + daysSinceCompetitionOpen + cOpen.sum3 + lPromo1 + rPromo1 + rnkState + xmas + winter + spring + summer + inPromo2 + newPromo2 + google2 + lPromo.wd + rPromo.wd + rOpen1 + lOpen1 + lOpen.sum30 + rOpen.sum30 + rHoliday1 + lHoliday1 + lHoliday.any3 + cHoliday.any15
  #config$xgb.dat.forumla = ~ year + month + day + as.numeric(DayOfWeek) + Promo + SchoolHoliday + Store + intStoreType + intAssortment + CompetitionDistance + Promo2 + as.numeric(PromoInterval) + as.numeric(CompetitionOpenSince) + as.numeric(Promo2Since) - 1 + CompetitionDistance2 + rnkStoreSales + rnkStoreCust + rnkStoreSPC + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + custPC1 + custPC2 + custPC3 + spcPC1 + spcPC2 + spcPC3 + google + Event + cOpen.sum7 + lHoliday.any7 + Max_TemperatureC + Min_TemperatureC + Precipitationmm + StateHoliday + AfterRefurb + DaysSinceOpening + daysSinceCompetitionOpen + cOpen.sum3 + lPromo1 + rPromo1 + rnkState + xmas + winter + spring + summer + inPromo2 + newPromo2 + google2 + lPromo.wd + rPromo.wd + rOpen1 + lOpen1 + lOpen.sum30 + rOpen.sum30 + rHoliday1 + lHoliday1 + lHoliday.any3 + cHoliday.any15 + CSIfood2
  
  #if (!is.null(config$lag.days) && config$lag.days <= 16) {
  #  cat('NOTE: Adding lagged TS features\n')
  #  config$xgb.dat.forumla = update.formula(config$xgb.dat.forumla, ~ . + lag.CustomersThisWeek + lag.CustomersLastWeek)
  #}

  # [now trying: additional pcs after successive regression]

  config$xgb.dat.forumla = ~ - 1 + 
    # date features
    year + month + day + as.integer(DayOfWeek) +
    #datePC1 + datePC2 + datePC3 + datePC4 + datePC5 + #datePC6 + datePC7 + datePC8 + datePC9 + datePC10 + datePC11 + datePC12 + datePC13 + datePC14 + datePC15 + 
    # store features
    Store + intStoreType + intAssortment + CompetitionDistance + Promo2 + as.integer(PromoInterval) + as.integer(CompetitionOpenSince) + as.integer(Promo2Since) + 
    storePC1 + storePC2 + storePC3 + storePC4 + storePC5 + storePC6 + storePC7 + storePC8 + storePC9 + storePC10 + #storePC11 + storePC12 + storePC13 + storePC14 + storePC15 + storePC16 + storePC17 + storePC18 + storePC19 + storePC20 + storePC21 + storePC22 + storePC23 + storePC24 + storePC25 + storePC26 + storePC27 + storePC28 + storePC29 + storePC30 + 
    # date x store features
    Promo + SchoolHoliday + as.integer(StateHoliday) + CompetitionDistance2 +
    Event + rEvent1 + lEvent1 + Precipitationmm + Max_TemperatureC + Min_TemperatureC + google + CSIfood2 + 
    AfterRefurb + DaysSinceOpening + daysSinceCompetitionOpen + inPromo2 + newPromo2 +
    lPromo1 + rPromo1 + #lPromo.wd + rPromo.wd + 
    rOpen1 + lOpen1 + cOpen.sum3 + cOpen.sum7 + lOpen.sum30 + rOpen.sum30 + 
    rHoliday1 + lHoliday1 + lHoliday.any3 + lHoliday.any7 + cHoliday.any15
  
  #config$xgb.dat.forumla = ~ -1 + year + month + day + as.numeric(DayOfWeek) +
  #  Event + rEvent1 + lEvent1 + Precipitationmm + Max_TemperatureC + Min_TemperatureC + google2 + CSIfood2 + #google
  #  lag.Sales1 + lag.Sales2 + lag.Sales3 + lag.Sales4 + lag.Sales5 + lag.Sales6 + lag.Sales7 +
  #  lag.Sales.wp + lag.Sales.wp.med4 + lag.Sales.wp.med12 + lag.SPC.mta + 
  #  lag.CustomersThisWeek + lag.CustomersLastWeek +
  #  lag.datePC1 + lag.datePC2 + lag.datePC3 +
  #  SchoolHoliday + StateHoliday + 
  #  Promo + lPromo1 + rPromo1 + lPromo.wd + rPromo.wd + 
  #  rOpen1 + lOpen1 + cOpen.sum3 + cOpen.sum7 + lOpen.sum30 + rOpen.sum30 + 
  #  rHoliday1 + lHoliday1 + lHoliday.any3 + lHoliday.any7 + cHoliday.any15
    
  #config$xgb.n.rounds = 10000
  #config$xgb.tuned.pars$eta = 0.05
  #config$xgb.tuned.pars$subsample = 0.7
  #config$xgb.tuned.pars$colsample_bytree = 0.8
  #config$xgb.tuned.pars$max_depth = 15

  # Override modes
  if (config$debug.model) {
    cat('NOTE: debugging => simplified xgb model\n')
    config$xgb.tuned.pars$max_depth = 10
    config$xgb.n.rounds = 100
  } else if (config$bagging.mode) {
    #cat('NOTE: bagging => aggressive xgb model\n')
    #config$xgb.tuned.pars$eta = 1
    #config$xgb.tuned.pars$max_depth = 500
    #config$xgb.tuned.pars$min_child_weight = 0
    #config$xgb.n.rounds = 5
    #config$xgb.early.stop.round = NULL
    #config$xgb.print.every.n = 1
  } else if (config$tunning.mode) {
    cat('NOTE: In tune mode. Current setup:\n')
    print(config$hp.setup)
    
    config$xgb.n.rounds = config$hp.setup$n.rounds
    config$xgb.tuned.pars$eta = config$hp.setup$eta
    config$xgb.tuned.pars$subsample = config$hp.setup$subsample
    config$xgb.tuned.pars$colsample_bytree = config$hp.setup$colsample_bytree
    config$xgb.tuned.pars$max_depth = config$hp.setup$max_depth
  }

  ############################
  # DIRTY EXPERIMENT: Format the data for XGB
  # Do this once for all the data, to make sure train/test are handled the same way.
  # For historical reasons and lazyness I'll just duplicate the data (tons of memory!)
  # FIXME support 'none' validation mode  
  
  cat(date(), 'DIRTY EXPERIMENT: transforming data to xgb format\n')
  
  data = rbind(dsets$train, dsets$calib, dsets$valid, dsets$test)
  tr.idx = 1:nrow(dsets$train)
  ca.idx = tail(tr.idx, n = 1) + (1:nrow(dsets$calib))
  va.idx = tail(ca.idx, n = 1) + (1:nrow(dsets$valid))
  te.idx = tail(va.idx, n = 1) + (1:nrow(dsets$test ))
  
  if (0) {
    # The "usual" way to do it:
    data = model.matrix(config$xgb.dat.forumla, data = data)
  } else if (1) {
    # This codes all factors with one-hot (rather than R's non-redundant coding)
    data = predict(dummyVars(config$xgb.dat.forumla, data = data), newdata = data)
  } else {
    # This saves a ton of RAM, naturally, but according to the doc this forces 
    # XGB to treat all 0s as missing because it ignores the 'missing' parameter
    # in xgb.DMatrix below??
    data = sparse.model.matrix(config$xgb.dat.forumla, data = data)
  }
  
  if (config$take.log) {
    xgb.trainset = xgb.DMatrix(data[tr.idx, ], missing = config$recode.na.as, label = dsets$train$logSales)
    xgb.calibset = xgb.DMatrix(data[ca.idx, ], missing = config$recode.na.as, label = dsets$calib$logSales)
    xgb.validset = xgb.DMatrix(data[va.idx, ], missing = config$recode.na.as, label = dsets$valid$logSales)
  } else {
    xgb.trainset = xgb.DMatrix(data[tr.idx, ], missing = config$recode.na.as, label = dsets$train$Sales) # FIXME can actually add the weights here...
    xgb.calibset = xgb.DMatrix(data[ca.idx, ], missing = config$recode.na.as, label = dsets$calib$Sales)
    xgb.validset = xgb.DMatrix(data[va.idx, ], missing = config$recode.na.as, label = dsets$valid$Sales)
  }
  xgb.testset = xgb.DMatrix(data[te.idx, ], missing = config$recode.na.as)
  xgb.DMatrix.save(xgb.trainset, 'xgb-trainset.data')
  xgb.DMatrix.save(xgb.calibset, 'xgb-calibset.data')
  xgb.DMatrix.save(xgb.validset, 'xgb-validset.data')
  xgb.DMatrix.save(xgb.testset , 'xgb-testset.data' )
  ############################
  
  return (config)
}

config$shutdown.xgb = function(config) {
  return (config)
}

config$train.xgb = function(config, trainset, validset = NULL) {
  # DIRTY EXPERIMENT: part of the above experiment about encoding the data in xgb format
  xgb.trainset = xgb.DMatrix('xgb-trainset.data')
  xgb.validset = xgb.DMatrix('xgb-validset.data')
  watchlist = list(valid = xgb.validset, train = xgb.trainset)

  x.mod.t = xgb.train(print.every.n = config$xgb.print.every.n,
    params = c(config$xgb.fixed.pars, config$xgb.tuned.pars, seed = config$rng.seed), 
    nrounds = config$xgb.n.rounds, watchlist = watchlist, early.stop.round = config$xgb.early.stop.round, maximize = F,
    data = xgb.trainset, nthread = ifelse(config$compute.backend != 'serial', 1, config$nr.threads)
  )
  
  if (length(watchlist) == 2 && !is.null(config$xgb.early.stop.round)) {
    cat('\n', date(), 'Best validation result:', x.mod.t$bestScore, '@ round', x.mod.t$bestInd, '\n')
  }

  if (config$show.feature.importance && config$xgb.fixed.pars$booster != 'gblinear') {
    # Examine variable importance (takes some time to extract!)
    cat(date(), 'Examining importance of features in the single XGB model\n')
    impo = xgb.importance(colnames(predict(dummyVars(config$xgb.dat.forumla, data = trainset), newdata = trainset)), model = x.mod.t)
    #xgb.plot.importance(impo)
    print(impo[1:50, ])
  }
  
  return (x.mod.t)
} 

config$predict.xgb = function(config, model, dataset) {
  # DIRTY EXPERIMENT
  if (config$predict.dataset == 'train') {
    xgb.dataset = xgb.DMatrix('xgb-trainset.data')
  } else if (config$predict.dataset == 'calib') {
    xgb.dataset = xgb.DMatrix('xgb-calibset.data')
  } else if (config$predict.dataset == 'valid') {
    xgb.dataset = xgb.DMatrix('xgb-validset.data')
  } else {
    xgb.dataset = xgb.DMatrix('xgb-testset.data')
  }
  data.preds = predict(model, xgb.dataset)
  return (data.preds)
}

config$tune.xgb = function(config, trainset) {
  if (config$take.log) {
    trainset.y = trainset$logSales
  } else {
    trainset.y = trainset$Sales
  }
  
  xgb.trainset = config$as.xgb.DMatrix(config, trainset, trainset.y)

  # FIXME using K-fold CV, even if this is highly biased due to the temporal correlations...
  # Can stratify to have similar number or validation examples from each store.
  # Can do rolling validation (but will have to sacrifice training set size...)
  xgb.cv.res = xgb.cv(params = c(config$xgb.fixed.pars, config$xgb.tuned.pars, seed = config$rng.seed), data = xgb.trainset, nrounds = config$xgb.n.rounds, nfold = 5)
  
  plot (1:xgb.n.rounds, xgb.cv.res$train.rmspe.mean, type = 'l', lty = 2, ylab = 'RMSPE', xlab = 'Boosting round', main = 'Naive CV')
  lines(1:xgb.n.rounds, xgb.cv.res$test.rmspe.mean + 2 * xgb.cv.res$test.rmspe.std, lty = 3)
  lines(1:xgb.n.rounds, xgb.cv.res$test.rmspe.mean)
  lines(1:xgb.n.rounds, xgb.cv.res$test.rmspe.mean - 2 * xgb.cv.res$test.rmspe.std, lty = 3)
  abline(h = 0.09, col = 2)
  abline(h = 0.10, col = 'orange')
  abline(h = 0.12, col = 3)
}

if (config$use.model == 'xgb') {
  config$train    = config$train.xgb
  config$predict0 = config$predict.xgb
  config$tune     = config$tune.xgb
  config$init     = config$init.xgb
  config$shutdown = config$shutdown.xgb
} else if (config$use.model == 'ts') {
  config$train    = config$train.ts
  config$predict0 = config$predict.ts
  config$tune     = config$tune.ts
  config$init     = config$init.ts
  config$shutdown = config$shutdown.ts
} else if (config$use.model == 'storewise') {
  config$train    = config$train.ts
  config$predict0 = config$predict.storewise
  config$tune     = config$tune.ts
  config$init     = config$init.ts
  config$shutdown = config$shutdown.ts
} else {
  stop('Unexpected model')
}

config$predict = function(config, model, dataset) {
  preds = config$predict0(config, model, dataset)

  if (config$take.log) {
    preds = expm1(preds)
  }
  
  if (0) {
    # this should have no effect, since I removed all these from the trainset, 
    # and the Kaggle evaluation removes them from the dataset (unless there are
    # errors in which Open == 0 but Sales > 0...)
    preds[dataset$Open == 0] = 0
  }
  
  return (preds)
}

config$eval.valid.error = function(config, valid.preds) {
  valid.errs = ((valid.preds$predSales - valid.preds$Sales) / valid.preds$Sales) ^ 2
  valid.errs = valid.errs[valid.preds$Sales != 0]
  #plot(ecdf(valid.errs), xlim = c(0, 0.3), main = 'Validation erros eCDF')
  valid.RMSPE = sqrt(mean(valid.errs))
  valid.RMSPE.se = sd(valid.errs) / sqrt(length(valid.errs)) # this might not be appropriate
  valid.RMSPE.lo = valid.RMSPE - 2 * valid.RMSPE.se
  valid.RMSPE.hi = valid.RMSPE + 2 * valid.RMSPE.se
  return (list(valid.RMSPE = valid.RMSPE, valid.RMSPE.lo = valid.RMSPE.lo, valid.RMSPE.hi = valid.RMSPE.hi))
}

config$calibrate = function(config, calib, test) {
  calib[, uncalib.predSales := predSales]
  test [, uncalib.predSales := predSales]

  cat(date(), 'Calibrating...\n')
  uncalib.err = config$eval.valid.error(config, calib)$valid.RMSPE
  
  if (config$calibration.scheme == 'best') {
    # I submitted manually a few times to optimize (on the public LB) the slope of a linear 
    # transform of the predicted sales. The hope is that this modest tuning does more good than 
    # overfitting...
    sbmt.best = read.csv('rossm-submit-curr-best.csv')
    pub.lb.ids = test$Id[(test$Date <= min(test$Date) + round(0.4 * 48))] # Pub LB dates, I think
    # calib is probably irrelevant here, so don't touch it
    scl = exp(mean(log(sbmt.best$Sales[sbmt.best$Id %in% pub.lb.ids])) - mean(log(test$predSales[test$Id %in% pub.lb.ids])))
    cat('Multiplying test preds by', scl, '\n')
    test$predSales  = test$predSales * scl
  } else if (config$calibration.scheme == '0.985') {
    # For some justification, see 
    # https://www.kaggle.com/c/rossmann-store-sales/forums/t/17601/correcting-log-sales-prediction-for-rmspe
    calib$predSales = calib$predSales * 0.985
    test$predSales  = test$predSales  * 0.985
  } else if (config$calibration.scheme == 'linear') { # Linear (positive, but no need to constrain)
    lm.fit = lm(Sales ~ predSales - 1, calib, weights = 1 / calib$Sales ^ 2)
    cat(date(), 'Fitted linear model:', coef(lm.fit), '\n')
    #print(summary(lm.fit))
    calib$predSales = fitted(lm.fit)
    #plot(calib$uncalib.predSales, calib$predSales)
    #abline(0, 1, col = 2)
    test$predSales = predict(lm.fit, test)
  } else if (config$calibration.scheme == 'affine') { # Affine
    lm.fit = lm(Sales ~ predSales, calib, weights = 1 / calib$Sales ^ 2)
    cat(date(), 'Fitted affine model:', coef(lm.fit), '\n')
    #print(summary(lm.fit))
    calib$predSales = fitted(lm.fit)
    #plot(calib$uncalib.predSales, calib$predSales)
    #abline(0, 1, col = 2)
    test$predSales = predict(lm.fit, test)
  } else if (config$calibration.scheme == 'robust') { # robust linear
    # clearly there are outliers! we can't do anything about those, so robustify
    library(MASS)
    rlm.fit = rlm(Sales ~ predSales - 1, calib, weights = 1 / calib$Sales ^ 2)
    cat(date(), 'Fitted robust linear model:', coef(rlm.fit), '\n')
    #print(summary(rlm.fit))
    calib$predSales = fitted(rlm.fit)
    #plot(calib$uncalib.predSales, calib$predSales)
    #abline(0, 1, col = 2)
    test$predSales = predict(rlm.fit, test)
  } else if (config$calibration.scheme == 'spline') { # Smoothing spline
    sp.fit = smooth.spline(calib$predSales, calib$Sales, w = 1 / calib$Sales ^ 2, df = 10)
    #sp.fit = smooth.spline(calib$predSales, calib$Sales, w = 1 / calib$Sales ^ 2) # CV used for selecting DF
    #plot(sp.fit)
    #abline(0, 1, col = 2)
    calib$predSales = fitted(sp.fit)
    calib$predSales[is.na(calib$predSales)] = calib$uncalib.predSales[is.na(calib$predSales)] # ?!
    test$predSales = predict(sp.fit, test$predSales)$y
  } else if (config$calibration.scheme == 'isoreg') { # Simple isotonic regression 
    # It is pointless here since the x's are noisy, so
    # imposing strinct isotonicity constraints make no sense
    if (0) {
      # Use stats
      fit = isoreg(calib$predSales, calib$Sales)
      #plot(fit)
      calib$predSales[order(calib$predSales)] = exp(fit$yf)
      test$predSales = (as.stepfun(fit))(test$predSales)
    } else if (0) {
      # Use Iso
      library(Iso)
      fit = pava(calib$Sales[order(calib$predSales)], calib$Sales ^ (-2), stepfun = T)
      test$predSales = fit(test$predSales)
    } else {
      # Use mine
      source('isotonic-regression.r')
      fit = iam.train(calib$predSales, calib$Sales)
      test$predSales = iam.predict(fit, test$predSales)
    }
    plot(test$uncalib.predSales, test$predSales)
    abline(0, 1, col = 2)
  } else if (config$calibration.scheme == 'piso') { # penalized isotonic 
    # would be nice, but the implementation can't handle data of this magnitude (~40K samples, 1D....)
    library(isotonic.pen)
    n = length(calib$Sales) # the package has a bug so it needs this...
    fit = iso_pen(calib$Sales, calib$predSales, wt = 1 / calib$Sales ^ 2)
    # ...never happens...
    test$predSales = predict(fit, test$predSales)
  } else if (config$calibration.scheme == 'nir') { # Nearly isotonic regression
    library(neariso)
    neariso(calib$Sales, order(calib$predSales)) # whaddabudeweights?
    # TODO haven't tried it yet, does it make sense here?
  }
  
  calib.err = config$eval.valid.error(config, calib)$valid.RMSPE
  cat(date(), 'Uncalibrated calibration-set RMSPE =', uncalib.err, 'calibrated RMSPE =', calib.err, '\n')
  
  return (test)
}

config$undo.calibrate = function(config, test) {
  test$predSales = test$uncalib.predSales
  return (test)
}

config$pipeline = function(config, pp.data) {
  # Split the data and do final cleaning
  dsets = config$preprocess3(config, pp.data)
  rm(pp.data)
  
  # Initialize model
  config = config$init(config, dsets)
  
  if (config$use.model != 'ts' && config$hide.validation) {
    stopifnot(config$validation.scheme != 'random') # this is not handled here yet
    valid.date1 = min(dsets$valid$Date) - 1 + config$lag.days
    cat('NOTE: Validation performance will only be for dates <=', as.character(valid.date1), '\n')
    valid.idx = dsets$valid$Date <= valid.date1
  } else if (config$validation.scheme != 'none') {
    valid.idx = rep(T, nrow(dsets$valid))
  }

  if (config$validation.scheme != 'none') {
    model = config$train(config, dsets$train, dsets$valid[valid.idx, ])
  } else {
    model = config$train(config, dsets$train)
  }
  
  if (config$validation.scheme != 'none') {
    cat(date(), 'Validating...\n')
    
    config$predict.dataset = 'valid'
    preds = config$predict(config, model, dsets$valid)
    valid.preds = dsets$valid[, .(Id, Date, Open, Sales)]
    valid.preds[, predSales := preds]

    valid.err = config$eval.valid.error(config, valid.preds[valid.idx, ])
    cat(date(), 'RMSPE =', valid.err$valid.RMSPE, '[', valid.err$valid.RMSPE.lo, '...', valid.err$valid.RMSPE.hi, ']', '\n')
  } else {
    valid.preds = NULL
    valid.err = NULL
  }
  
  if (config$calibration.scheme != 'none') {
    cat(date(), 'Generating calibration preds...\n')
    
    config$predict.dataset = 'calib'
    preds = config$predict(config, model, dsets$calib)
    calib.preds = dsets$calib[, .(Id, Date, Open, Sales)]
    calib.preds[, predSales := preds]
  } else {
    calib.preds = NULL
  }
  
  # Generate test predictions
  config$predict.dataset = 'test'
  preds = config$predict(config, model, dsets$test)
  test.preds = dsets$test[, .(Id, Date, Open)]
  test.preds[, predSales := preds]

  config = config$shutdown(config)
  
  return (list(test = test.preds, valid = valid.preds, valid.err = valid.err, calib = calib.preds))
}

# Preprocessing and pipeline, for a single target day
config$pipeline.day = function(config, lag.days) {
  config$lag.days = lag.days
  test.date = as.Date('2015-07-31') + config$lag.days
  cat(date(), 'Working on lag.days =', config$lag.days, '( i.e., test.date =', as.character(test.date), ')', '\n')

  pp1.data.fn = sprintf('%s/ross-data1-%s.RData', config$project.dir, config$data.tag)
  pp2.data.fn = sprintf('%s/ross-data2-lag%g.RData', config$project.dir, config$lag.days)
  
  cat('Loading previously preprocessed data from', pp1.data.fn, '\n')
  load(file = pp1.data.fn) # => pp1.config, pp1.data
  config$valid.truth = pp1.config$valid.truth
  rm(pp1.config)
  
  if (config$do.preprocess2) {
    cat(date(), 'Generating lagged response features\n')
    pp2.config = config
    pp2.data = config$preprocess2(config, pp1.data)

    cat(date(), 'Saving lagged response features to', pp2.data.fn, '\n')
    save(pp2.config, pp2.data, file = pp2.data.fn)
  } else {
    cat(date(), 'Loading existing data for lagged response features from', pp2.data.fn, '\n')
    load(file = pp2.data.fn) # => pp2.config, pp2.data
  }
  rm(pp2.config)
  
  stopifnot(all(pp1.data$Id == pp2.data$Id)) # both should include all data, and be sorted the same way
  pp2.data[, Id := NULL]
  pp.data = cbind(pp1.data, pp2.data)
  rm(pp1.data, pp2.data)
  
  preds.single = config$pipeline(config, pp.data)
  
  if (config$tunning.mode) {
    return (preds.single$valid.err)
  } else {
    preds.single.config = config
    if (config$bagging.mode) {
      preds.fn = sprintf('%s/ross-preds-lag%g-bag%g.RData', config$project.dir, config$lag.days, config$bag.number)
    } else {
      preds.fn = sprintf('%s/ross-preds-lag%g.RData', config$project.dir, config$lag.days)
    }
    cat(date(), 'Saving predictions to', preds.fn, '\n')
    save(preds.single.config, preds.single, file = preds.fn)
    rm(preds.single.config)
    return (1)
  }
}

config$generate.submission = function(config, test.preds) {
  cat(date(), 'Generating submission\n')
  
  subfn = paste0('rossm-submit-', config$submit.tag, '.csv')
  modfn = paste0('rossm-submit-', config$submit.tag, '.RData')
  
  sbmt = data.frame(Id = test.preds$Id, Sales = test.preds$predSales)
  write.csv(sbmt, subfn, row.names = F, quote = F)
  save(config, file = modfn)
  
  cat('Saved under', subfn, '\n')
  
  # Sanity check:  
  if (0) {
    # compare to a reasonable submission downloaded from 
    # Kaggle script: https://www.kaggle.com/abhilashawasthi/rossmann-store-sales/xgb-rossmann/run/86608/code
    # (public LB score 0.10361)
    ref.submission = read.csv('rossm-ref.csv')
  } else {
    # My best submission
    # (public LB score 0.09802)
    ref.submission = read.csv('rossm-submit-vld.csv')
  }
  names(ref.submission) = c('Id', 'RefSales')
  amisane = merge(test.preds, ref.submission, by = 'Id')
  idx = amisane$Open == 1
  pdf('rossm-ref-sanity.pdf')
  plot(amisane$RefSales[idx], amisane$predSales[idx])
  abline(0, 1, col = 2)
  dev.off()
}

blend = function() {
  # NOTE: I ran this manually
  
  if (0) {
    # Scaling my best submission so far, improves the result considerably
    scale.factor = 0.985
    # scale.factor => LB score when starting from rossm-submit-rev2.csv:
    # * 1     => 0.09778
    # * 0.99  => 0.09618
    # * 0.985 => 0.09578
    # * 0.975 => 0.09578
    
    sbmt = read.csv('rossm-submit-rev2.csv')
    sbmt$Sales = sbmt$Sales * scale.factor
    write.csv(sbmt, 'rossm-submit-rev2-scaled.csv', row.names = F, quote = F)
    
    sbmt.old = read.csv('rossm-submit-rev2.csv')
    plot(sbmt.old$Sales, sbmt$Sales)
    abline(0, 1, col = 2)
  }
  
  if (0) {
    sbmt.best = read.csv('rossm-submit-curr-best.csv')
    sbmt.new = read.csv('rossm-submit-cal.csv')
    mean(log(sbmt.best$Sales))
    mean(log(sbmt.new$Sales))
    sbmt.new$Sales = sbmt.new$Sales * exp(mean(log(sbmt.best$Sales)) - mean(log(sbmt.new$Sales)))
    mean(log(sbmt.new$Sales))
    plot(sbmt.best$Sales, sbmt.new$Sales, pch = '.')
    abline(0, 1, col = 2)
    write.csv(sbmt.new, 'rossm-submit-cal-recal.csv', row.names = F, quote = F)
  }
  
  if (0) {
    # Trying to scale my last bagged submission:
    scale.factor = 1.02
    # scale.factor => LB score when starting from rossm-submit-rev2.csv:
    # * 1     => 0.09636
    # * 0.99  => 0.09744
    # * 1.02  => 0.09724
    
    sbmt = read.csv('rossm-submit-bagf.csv')
    sbmt$Sales = sbmt$Sales * scale.factor
    write.csv(sbmt, 'rossm-submit-bagf-scaled.csv', row.names = F, quote = F)
    
    sbmt.old = read.csv('rossm-submit-bagf.csv')
    plot(sbmt.old$Sales, sbmt$Sales)
    abline(0, 1, col = 2)
  }
  
  if (0) {
    # Blend some of my submissions. Since all of them are pretty much the same, I don't expect to
    # get much out of it.
    
    sbmt1 = read.csv('rossm-submit-mono.csv')
    sbmt2 = read.csv('rossm-submit-bagf.csv')
    sbmt3 = read.csv('rossm-submit-cal2.csv')
    sbmt4 = read.csv('rossm-submit-rev2-scaled.csv')
    
    # Pub LB scores:
    # sbmt1: 0.09616
    # sbmt2: 0.09636
    # sbmt3: 0.09539
    # sbmt4: 0.09578
    
    # So let's weight them
    w1 = 0.2
    w2 = 0.2
    w3 = 0.3
    w4 = 0.3
    
    # Blend
    stopifnot(all(sbmt1$Id == sbmt2$Id & sbmt1$Id == sbmt3$Id & sbmt1$Id == sbmt3$Id & sbmt1$Id == sbmt4$Id))
    sbmt = sbmt1
    sbmt$Sales = w1 * sbmt1$Sales + w2 * sbmt2$Sales + w3 * sbmt3$Sales + w4 * sbmt4$Sales
    
    plot(sbmt1$Sales, sbmt$Sales)
    abline(0, 1, col = 2)
    
    write.csv(sbmt, 'rossm-submit-belnded.csv', row.names = F, quote = F)
    # (This achieved a Pub LB score of 0.09396)
  }
  
  if (0) {
    # Second blending attempt: blend more options, do it on the log scale
    s.to.blend = c('mono', 'bagf', 'cal2', 'rev2-scaled', 'mono2', 'mono3')
    w.to.blend = c(1     , 2     , 3     , 2            , 1      , 1      )
    
    w.to.blend = w.to.blend / sum(w.to.blend)
    
    for (i in 1:length(w.to.blend)) {
      cat('Adding', s.to.blend[i], 'with weight', w.to.blend[i], '\n')
      sbmt.new = read.csv(paste0('rossm-submit-', s.to.blend[i], '.csv'))
      sbmt.new$Sales = log1p(sbmt.new$Sales) * w.to.blend[i]
      if (i == 1) {
        sbmt.blend = sbmt.new
      } else {
        stopifnot(all(sbmt.blend$Id == sbmt.new$Id))
        sbmt.blend$Sales = sbmt.blend$Sales + sbmt.new$Sales
      }
    }
    sbmt.blend$Sales = expm1(sbmt.blend$Sales)
    
    # sanity check    
    sbmt.old = read.csv('rossm-submit-belnded.csv')
    plot(sbmt.old$Sales, sbmt.blend$Sales)
    abline(0, 1, col = 2)
    
    write.csv(sbmt.blend, 'rossm-submit-belnded2.csv', row.names = F, quote = F)
    # (This achieved a Pub LB score of 0.09271)
  }
  
  if (0) {
    # Third blending attempt: blend more options, do it on the log scale
    s.to.blend = c('bagn', 'bagf', 'cal2', 'rev2-scaled', 'mono', 'mono2', 'mono3', 'mono4', 'mono5', 'mono6', 'mono7', 'mono8', 'mono9')
    w.to.blend = c(2     , 2     , 3     , 3            , 1     , 1      , 1      , 1      , 1      , 1      , 1      , 1      , 1      )
    
    w.to.blend = w.to.blend / sum(w.to.blend)
    
    for (i in 1:length(w.to.blend)) {
      cat('Adding', s.to.blend[i], 'with weight', w.to.blend[i], '\n')
      sbmt.new = read.csv(paste0('rossm-submit-', s.to.blend[i], '.csv'))
      sbmt.new$Sales = log1p(sbmt.new$Sales) * w.to.blend[i]
      if (i == 1) {
        sbmt.blend = sbmt.new
      } else {
        stopifnot(all(sbmt.blend$Id == sbmt.new$Id))
        sbmt.blend$Sales = sbmt.blend$Sales + sbmt.new$Sales
      }
    }
    sbmt.blend$Sales = expm1(sbmt.blend$Sales)
    
    # sanity check    
    sbmt.old = read.csv('rossm-submit-belnded2.csv')
    plot(sbmt.old$Sales, sbmt.blend$Sales)
    abline(0, 1, col = 2)
    
    write.csv(sbmt.blend, 'rossm-submit-belnded3.csv', row.names = F, quote = F)
    # (This achieved a Pub LB score of 0.09212)
  }
  
  if (1) {
    # Fourth and final blending attempt: blend more options, do it on the log scale
    s.to.blend = c('bagn', 'bagf', 'cal2', 'rev2-scaled', 'mono', 'mono2', 'mono3', 'mono4', 'mono5', 'mono6', 'mono7', 'mono8', 'mono9')
    w.to.blend = c(5     , 1     , 3     , 3            , 1     , 1      , 1      , 1      , 1      , 1      , 1      , 1      , 1      )
    
    w.to.blend = w.to.blend / sum(w.to.blend)
    
    for (i in 1:length(w.to.blend)) {
      cat('Adding', s.to.blend[i], 'with weight', w.to.blend[i], '\n')
      sbmt.new = read.csv(paste0('rossm-submit-', s.to.blend[i], '.csv'))
      sbmt.new$Sales = log1p(sbmt.new$Sales) * w.to.blend[i]
      if (i == 1) {
        sbmt.blend = sbmt.new
      } else {
        stopifnot(all(sbmt.blend$Id == sbmt.new$Id))
        sbmt.blend$Sales = sbmt.blend$Sales + sbmt.new$Sales
      }
    }
    sbmt.blend$Sales = expm1(sbmt.blend$Sales)
    
    # sanity check    
    sbmt.old = read.csv('rossm-submit-belnded3.csv')
    plot(sbmt.old$Sales, sbmt.blend$Sales)
    abline(0, 1, col = 2)
    
    write.csv(sbmt.blend, 'rossm-submit-belnded4.csv', row.names = F, quote = F)
    # (This achieved a Pub LB score of 0.09213)
  }
}

# Do stuff
# ==============================================================================

if (config$do.log) {
  suppressWarnings(sink()) # make sure no leftover sinks
  sink(file = paste0('./tmp/ross-', strftime(Sys.time(), format = '%F_%H-%M-%S'), '.log'), split = T)
}

set.seed(config$rng.seed)

if (config$do.preprocess1) {
  cat(date(), 'Preprocessing, stage 1\n')
  pp1 = config$preprocess1(config)
  config$valid.truth = pp1$valid.truth
  pp1.data = pp1$data
  pp1.config = config
  save(pp1.config, pp1.data, file = sprintf('ross-data1-%s.RData', config$data.tag))
  rm(pp1.config)
} else if (config$do.pipeline && config$use.model == 'ts') {
  cat('NOTE: Loading previously preprocessed data\n')
  load(sprintf('ross-data1-%s.RData', config$data.tag)) # => pp1.config, pp1.data
  config$valid.truth = pp1.config$valid.truth
  rm(pp1.config)
}

if (config$do.pipeline) {
  if (config$use.model == 'ts') {
    # One pipeline fits all
    test.preds = config$pipeline(config, pp1.data)$test
    config.test.preds = config
    save(config.test.preds, test.preds, file = 'ross-preds-ts.RData')
    rm(config.test.preds)
  } else {
    if (config$single.mode) {
      cat('NOTE: In single mode. Launching pipe on day', config$single.mode.day, '\n')
      config$pipeline.day(config, config$single.mode.day)
    } else if (config$bagging.mode) {
      cat('NOTE: Launching bagging mode with lag', config$single.mode.day, '\n')
      
      bag.job = function(config, core) {
        config$bag.number = core
        config$pipeline.day(config, config$single.mode.day)
        return (1)
      }
      
      # (this generates output files, res is empty and ignored)
      res = compute.backend.run(
        config, bag.job, combine = rbind, 
        package.dependencies = config$package.dependencies,
        source.dependencies  = config$source.dependencies,
        cluster.dependencies = config$cluster.dependencies,
        cluster.batch.name = 'Kaggle', 
        cluster.requirements = config$cluster.requirements
      )
    } else {
      # The pipeline forks per test day
      # (this generates output files, res is empty and ignored)
      cat(date(), 'Launching pipeline on all lags\n')

      roll.job = function(config, core) {
        lag.days = core
        if (lag.days %in% config$rolling.days) {
          config$pipeline.day(config, lag.days)
        }
        return (1)
      }
      
      config$nr.cores = max(config$rolling.days)
      
      res = compute.backend.run(
        config, roll.job, combine = c, 
        package.dependencies = config$package.dependencies,
        source.dependencies  = config$source.dependencies,
        cluster.dependencies = config$cluster.dependencies,
        cluster.batch.name = 'Kaggle', 
        cluster.requirements = config$cluster.requirements
      )
    }
  }
}

if (config$do.submission) {
  cat(date(), 'Collecting predictions\n')
  
  if (config$use.model == 'ts') {
    save(file = 'ross-preds-ts.RData') # => config.test.preds, test.preds
    rm(config.test.preds)
  } else {
    if (config$bagging.mode) {
      preds = NULL
      #bags = 1:config$nr.cores # bad term but nm
      bags = c(147, 157, 181, 163, 115, 139, 159, 143, 127, 153, 177, 179, 39, 89, 169, 175, 151, 417, 493, 375, 103, 225, 125, 171, 61, 137, 497, 167, 173, 245, 117, 165, 73, 413, 463, 461, 129, 411, 141)
      
      for (bagi in 1:length(bags)) {
        bag = bags[bagi]
        load(file = sprintf('ross-preds-lag%g-bag%g.RData', config$single.mode.day, bag)) # => preds.single.config, preds.single
        rm(preds.single.config)
        
        # NOTE: I assume here that pred.single is always ordered the same way: by 
        # the (Store, Date) ordering of the valid/testset
        
        if (is.null(preds)) {
          preds = preds.single
        } else {
          if (config$validation.scheme != 'none') {
            preds$valid$predSales = expm1((log1p(preds$valid$predSales) * (bagi - 1) + log1p(preds.single$valid$predSales)) / bagi)
          }
          if (config$calibration.scheme != 'none') {
            preds$calib$predSales = expm1((log1p(preds$calib$predSales) * (bagi - 1) + log1p(preds.single$calib$predSales)) / bagi)
          }
          preds$test$predSales = expm1((log1p(preds$test$predSales) * (bagi - 1) + log1p(preds.single$test$predSales)) / bagi)
        }
        
        if (config$validation.scheme != 'none') {
          vld = config$eval.valid.error(config, preds$valid)
          cat('After processing', bagi, 'bags: validation RMSPE =', vld$valid.RMSPE, '[', vld$valid.RMSPE.lo, '...', vld$valid.RMSPE.hi, ']', '\n')
        } else {
          cat('Processed bag', bagi, '\n')
        }
      }
    } else {
      if (config$single.mode) {
        days.to.run = config$single.mode.day
        is.rolling.validation = F
      } else {
        days.to.run = sort(config$rolling.days, decreasing = T) # any model with lag L is applicable to days 1:L
        is.rolling.validation = (config$validation.scheme %in% c('last', 'stoypy'))
      }
      preds = NULL
  
      for (lag.days in days.to.run) {
        test.date = as.Date('2015-07-31') + lag.days
        load(file = sprintf('ross-preds-lag%g.RData', lag.days)) # => preds.single.config, preds.single
        rm(preds.single.config)
  
        # NOTE: I assume here that pred.single is always ordered the same way: by 
        # the (Store, Date) ordering of the valid/testset
        
        if (is.null(preds)) {
          preds = preds.single
          if (config$validation.scheme != 'none') {
            vldsz.randperm = sample(nrow(preds$valid))
          }
        } else {
          preds$test[Date <= test.date, ] = preds.single$test[Date <= test.date, ]
          
          if (is.rolling.validation) {
            valid.date = min(preds.single$valid$Date) - 1 + lag.days
            preds$valid[Date <= valid.date, ] = preds.single$valid[Date <= valid.date, ]
            # FIXME what do I do with calibration here?
          } else if (config$validation.scheme != 'none') {
            # There is no accurate way of doing this in general, so just wing it
            idxs = vldsz.randperm[1:round(lag.days / 48 * nrow(preds$valid))]
            preds$valid[idxs, ] = preds.single$valid[idxs, ]
          }
        }
        
        if (config$validation.scheme != 'none') {
          vld = config$eval.valid.error(config, preds$valid)
          cat('After processing lag', lag.days, ': validation RMSPE =', vld$valid.RMSPE, '[', vld$valid.RMSPE.lo, '...', vld$valid.RMSPE.hi, ']', '\n')
        }
      }
    }

    if (config$calibration.scheme != 'none') {
      preds$test = config$calibrate(config, preds$calib, preds$test)
    }
    
    test.preds = preds$test
  }
  
  config$generate.submission(config, test.preds)
}

if (config$tunning.mode) {
  cat('DBG: Tuning model using day 48\n')
  
  # 1. Define the hyperparameter grid
  if (1) {
    n.rounds = 5000
    eta = c(0.03, 0.025, 0.02, 0.015, 0.01)
    subsample = c(1, 0.9, 0.8, 0.7)
    colsample_bytree = c(0.8, 0.7, 0.6)
    max_depth = 8:15
  } else {
    n.rounds = 10
    eta = 0.05
    subsample = 0.9
    colsample_bytree = 1
    max_depth = 2:3
  }
  
  config$hp.grid = expand.grid(n.rounds = n.rounds, eta = eta, subsample = subsample, colsample_bytree = colsample_bytree, max_depth = max_depth)
  config$nr.cores = nrow(config$hp.grid)

  # 2. Define a function that sets up a single grid point and calls pipeline.day
  tune.job = function(config, core) {
    config$hp.setup = config$hp.grid[core, ]
    valid.err = config$pipeline.day(config, 48) # NOTE: hard coded to tuning the monolithic model
    return (cbind(config$hp.setup, valid.RMSPE = valid.err$valid.RMSPE))
  }
  
  # 3. Call it through ComputeBackend
  cat(date(), 'Launching cluster run\n')
  res = compute.backend.run(
    config, tune.job, combine = rbind, 
    package.dependencies = config$package.dependencies,
    source.dependencies  = config$source.dependencies,
    cluster.dependencies = config$cluster.dependencies,
    cluster.batch.name = 'Kaggle', 
    cluster.requirements = config$cluster.requirements
  )

  save(res, file = 'rossm-tune.RData')
  
  cat(date(), '\nTuning results:\n\n')
  res = res[order(res$valid.RMSPE), ]
  print(res)
  cat('\n')
}

cat(date(), 'Done.\n')

if (config$do.log) {
  sink()
}
