/*
Inspired by https://github.com/jinglescode/time-series-forecasting-tensorflowjs
 */

let features = [];
let featuresDates = [];
let labels = [];
let labelsDates = [];

let plotLayout = {margin: {t: 0, r: 0}, legend: {orientation: "h", x: 0, y: 1}};

// let label_col = 'AAPL.Close';
// let date_col = 'Date';
let non_overlapping = true;
let round_digits = 4;
let horizon;
let window_size;
let train_prop;
let n_epochs;
let learning_rate;
let n_layers;


let forecastTableId = '#forecast_table';
let dataTableId = '#data_table';
let metricsTableId = '#metrics_table';
let labelColInput = document.getElementById('label_col');
let dateColInput = document.getElementById('date_col');
let divLog = document.getElementById("div_log")
let divTrainLog = document.getElementById("div_train_log");
let divPlotLoss = document.getElementById('div_plot_loss');
let divPlotTrainLoss = document.getElementById('div_plot_train_loss');
let divPlotTrainForecast = document.getElementById('div_plot_train_forecast');
let divPlotForecast = document.getElementById('div_plot_forecast');
let submitButton = document.getElementById('submit_button');

let trainLosses = []
let valLosses = []

var rawData;
var values;
var dates;


function activateTab(tab) {
    $('.nav-pills a[href="#' + tab + '"]').tab('show');
};


function clearResults() {
    divLog.innerHTML = ''
    divTrainLog.innerHTML = ''

    divPlotLoss.innerHTML = ''
    divPlotTrainLoss.innerHTML = ''

    divPlotTrainForecast.innerHTML = ''
    divPlotForecast.innerHTML = ''

    $(forecastTableId + ' thead').empty()
    $(forecastTableId + ' tbody').empty()

    $(metricsTableId + ' thead').empty()
    $(metricsTableId + ' tbody').empty()
}

function setLoadingMessage() {
    submitButton.innerHTML = 'Running...'
    submitButton.disabled = true;
}

function removeLoadingMessage() {
    submitButton.innerHTML = 'Train model'
    submitButton.disabled = false;
}


function writeTable(tableId, tableData, skipFirstHeading = false) {
    let obj

    let thead = $(tableId + ' thead')
    obj = tableData[0]
    let tr = "";
    let i = 0
    for (let [key, value] of Object.entries(obj)) {
        if (!skipFirstHeading || i !== 0) {
            tr += "<td>" + key + "</td>"
        } else {
            tr += "<td></td>"
        }
        i++;
    }
    tr += ""
    thead.append(tr);

    let tbody = $(tableId + ' tbody')
    for (let i = 0; i < tableData.length; i++) {
        obj = tableData[i]
        let tr = "<tr>";
        for (let [key, value] of Object.entries(obj)) {
            tr += "<td>" + value + "</td>"
        }
        tbody.append(tr);
    }
}


async function fit(X, Y, callback, args, X_val, y_val) {

    const batch_size = 32;

    // input dense layer
    const input_layer_shape = args.window_size;
    const input_layer_neurons = 64;

    // LSTM
    const rnn_input_layer_features = 16;
    const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;
    const rnn_input_shape = [rnn_input_layer_features, rnn_input_layer_timesteps];
    const rnn_output_neurons = 16;

    // Output dense layer
    const output_layer_shape = rnn_output_neurons;
    const output_layer_neurons = args.horizon;

    // Load into tensor and normalize data
    const X_tensor = tf.tensor2d(X, [X.length, X[0].length]);
    const y_tensor = tf.tensor2d(Y, [Y.length, Y[0].length]);

    const [xs, inputMax, inputMin] = normalizeTensorFit(X_tensor);
    const [ys, labelMax, labelMin] = normalizeTensorFit(y_tensor);

    let validationData = null;
    if (X_val !== undefined && y_val !== undefined) {
        const X_test_tensor = tf.tensor2d(X_val, [X_val.length, X_val[0].length]);
        const y_test_tensor = tf.tensor2d(y_val, [y_val.length, y_val[0].length]);
        const xs_test = normalizeTensor(X_test_tensor, inputMax, inputMin);
        const ys_test = normalizeTensor(y_test_tensor, inputMax, inputMin);
        validationData = [xs_test, ys_test]
    }

    // Define model
    const model = tf.sequential();

    model.add(tf.layers.dense({units: input_layer_neurons, inputShape: [input_layer_shape]}));
    model.add(tf.layers.reshape({targetShape: rnn_input_shape}));

    let lstm_cells = [];
    for (let index = 0; index < args.n_layers; index++) {
        lstm_cells.push(tf.layers.lstmCell({units: rnn_output_neurons}));
    }

    model.add(tf.layers.rnn({
        cell: lstm_cells,
        inputShape: rnn_input_shape,
        returnSequences: false
    }));

    model.add(tf.layers.dense({units: output_layer_neurons, inputShape: [output_layer_shape]}));

    model.compile({
        optimizer: tf.train.adam(args.learning_rate),
        loss: 'meanSquaredError'
    });

    // fit model
    const hist = await model.fit(xs, ys,
        {
            batchSize: batch_size,
            epochs: args.n_epochs,
            validationData: validationData,
            callbacks: {
                onEpochEnd: async (epoch, log) => {
                    callback(epoch, log);
                }
            }
        });

    return {
        model: model,
        stats: hist,
        normalize: {inputMax: inputMax, inputMin: inputMin, labelMax: labelMax, labelMin: labelMin}
    };
}

function makePredictions(X, model, dict_normalize) {
    X = tf.tensor2d(X, [X.length, X[0].length]);
    const normalizedInput = normalizeTensor(X, dict_normalize["inputMax"], dict_normalize["inputMin"]);
    const model_out = model.predict(normalizedInput);
    const predictedResults = unNormalizeTensor(model_out, dict_normalize["labelMax"], dict_normalize["labelMin"]);

    return Array.from(predictedResults.dataSync());
}

function normalizeTensorFit(tensor) {
    const maxval = tensor.max();
    const minval = tensor.min();
    const normalizedTensor = normalizeTensor(tensor, maxval, minval);
    return [normalizedTensor, maxval, minval];
}

function normalizeTensor(tensor, maxval, minval) {
    return tensor.sub(minval).div(maxval.sub(minval));
}

function unNormalizeTensor(tensor, maxval, minval) {
    return tensor.mul(maxval.sub(minval)).add(minval);
}


let callback = function (epoch, log) {
    let logHtml = divLog.innerHTML;
    logHtml = `
       <div><small>Epoch: ${epoch + 1}/${n_epochs}, Loss: ${log.loss.toFixed(round_digits)}</small></div>
    ` + logHtml;

    trainLosses.push(log.loss);
    divLog.innerHTML = logHtml;
    if (epoch === 1 || ((n_epochs < 2) && ((epoch + 1) === n_epochs))) {
        activateTab('forecast');
    }

    Plotly.newPlot(divPlotLoss, [{
        x: Array.from({length: trainLosses.length}, (v, k) => k + 1),
        y: trainLosses,
        name: "Final model"
    }], plotLayout);
};

let trainCallback = function (epoch, log) {
    let logHtml = divTrainLog.innerHTML;
    logHtml = `
       <div><small>
       Epoch: ${epoch + 1}/${n_epochs},
       Train Loss: ${log.loss.toFixed(round_digits)},
       Val Loss: ${log.val_loss.toFixed(round_digits)}
       </small></div>
    ` + logHtml;

    trainLosses.push(log.loss);
    valLosses.push(log.val_loss);
    divTrainLog.innerHTML = logHtml;
    let xAxis = Array.from({length: trainLosses.length}, (v, k) => k + 1);
    Plotly.newPlot(divPlotTrainLoss, [
        {x: xAxis, y: trainLosses, name: "Train loss"},
        {x: xAxis, y: valLosses, name: "Val loss"},
    ], plotLayout);
};


async function prepareData(window_size, horizon) {
    if (rawData === undefined) {
        let errorMessage = 'Please upload a CSV file'
        alert(errorMessage)
        removeLoadingMessage()
        throw new Error(errorMessage);
    }

    let label_col = $('#label_col').val();
    if (label_col === 'NA') {
        let errorMessage = 'Please select the label column'
        alert(errorMessage)
        removeLoadingMessage()
        throw new Error(errorMessage);
    }
    values = rawData.map(r => parseFloat(r[label_col]));


    let date_col = $('#date_col').val();
    if (date_col === 'NA') {
        dates = Array.from({length: values.length}, (v, k) => k + 1)
    } else {
        dates = rawData.map(r => r[date_col]);
    }


    features = []
    featuresDates = []
    labels = []
    labelsDates = []
    let n_rows = values.length;
    let window_start, window_end, label_start, label_end,
        row_features, row_labels, row_features_dates, row_labels_dates;
    let loop_end = n_rows - window_size - horizon + 1

    let i = 0;
    while (i < loop_end) {
        window_start = i;
        window_end = i + window_size;
        label_start = window_end;
        label_end = label_start + horizon;

        row_features = values.slice(window_start, window_end)
        row_features_dates = dates.slice(window_start, window_end)
        row_labels = values.slice(label_start, label_end)
        row_labels_dates = dates.slice(label_start, label_end)

        features.push(row_features)
        featuresDates.push(row_features_dates)
        labels.push(row_labels)
        labelsDates.push(row_labels_dates)

        if (non_overlapping) {
            i += horizon;
        } else {
            i++;
        }
    }
}

$("#showAdvancedParams").click(function () {
    $("#showText").text(($("#showText").text() == 'Show') ? 'Hide' : 'Show');
})

async function errorMetrics(yTrue, yPred, label) {
    let yTrueT = tf.tensor(yTrue)
    let yPredT = tf.tensor(yPred)
    let rmse = await yTrueT.sub(yPredT).square().sum().div(tf.scalar(yPred.length)).sqrt().data()
    rmse = rmse[0]
    let mae = await yTrueT.sub(yPredT).abs().sum().div(tf.scalar(yPred.length)).data()
    mae = mae[0]
    // let mae = await yTrueT.sub(yPredT).square().sum().div(tf.scalar(yPred.length)).sqrt().data()[0]
    return {"": label, "RMSE": rmse.toFixed(round_digits), "MAE": mae.toFixed(round_digits)}
}

async function runAnalysis() {
    setLoadingMessage();
    clearResults();
    activateTab('train');

    horizon = parseInt($('#horizon').val());
    window_size = parseInt($('#window_size').val());
    train_prop = parseFloat($('#train_prop').val());
    n_epochs = parseInt($('#n_epochs').val());
    learning_rate = parseFloat($('#learning_rate').val());
    n_layers = parseInt($('#n_layers').val());

    await prepareData(window_size, horizon)

    // Val results
    let n = features.length
    let train_end = Math.floor(train_prop * n)

    let XTrain = features.slice(0, train_end)
    let yTrain = labels.slice(0, train_end)
    let datesTrain = labelsDates.slice(0, train_end)

    let XVal = features.slice(train_end, n)
    let yVal = labels.slice(train_end, n)
    let datesVal = labelsDates.slice(train_end, n)

    let trainArgs = {
        window_size: window_size,
        horizon: horizon,
        n_epochs: n_epochs,
        learning_rate: learning_rate,
        n_layers: n_layers,
    }
    trainLosses = []
    valLosses = []
    let modelTrain = await fit(features, labels, trainCallback, trainArgs, XVal, yVal);

    let predsTrain = makePredictions(XTrain, modelTrain['model'], modelTrain['normalize']);
    let trainMetrics = await errorMetrics(yTrain.flat(), predsTrain, 'Train')

    let predsVal = makePredictions(XVal, modelTrain['model'], modelTrain['normalize']);
    let valMetrics = await errorMetrics(yVal.flat(), predsVal, 'Val')
    writeTable(metricsTableId, [trainMetrics, valMetrics])

    let pointSize = 4;
    let lineWidth = 2;
    let actualsColor = "#1F77B4"
    let predsColor = "#FF7F0E"
    let mode = "lines+markers"

    Plotly.newPlot(divPlotTrainForecast, [
        {
            x: datesTrain.flat(), y: yTrain.flat(), name: "Actual (train)", mode: mode,
            marker: {size: pointSize, color: actualsColor}, line: {width: lineWidth, color: actualsColor},
        },
        {
            x: datesTrain.flat(), y: predsTrain, name: "Prediction (train)", mode: mode,
            marker: {size: pointSize, color: predsColor}, line: {width: lineWidth, color: predsColor}
        },
        {
            x: datesVal.flat(), y: yVal.flat(), name: "Actual (val)", mode: mode,
            marker: {size: pointSize, color: actualsColor}, line: {width: lineWidth, color: actualsColor},
        },
        {
            x: datesVal.flat(), y: predsVal, name: "Prediction (val)", mode: mode,
            marker: {size: pointSize, color: predsColor}, line: {width: lineWidth, color: predsColor},
        },
    ], {margin: {t: 0, r: 0}, legend: {orientation: "h", x: 0, y: 1}});


    // Full model
    trainLosses = []
    let model = await fit(features, labels, callback, trainArgs);

    // Out-of-sample forecasts
    let X_oos = [values.slice(-window_size)];
    let preds = makePredictions(X_oos, model['model'], model['normalize']);

    let predsTableValues = {}
    for (let i = 0; i < preds.length; i++) {
        predsTableValues[`T + ${i + 1}`] = preds[i].toFixed(round_digits)
    }
    writeTable(forecastTableId, [predsTableValues]);

    let plotHistory = 30
    let plotValues = values.slice(-plotHistory)

    let xAxis = Array.from({length: plotValues.length + horizon}, (v, k) => k + 1)

    let tickText = dates.slice(-plotHistory)
    tickText = tickText.concat(Array.from({length: horizon}, (v, k) => `T + ${k + 1}`))

    Plotly.newPlot(divPlotForecast, [
            {
                x: xAxis.slice(0, plotValues.length), y: plotValues, name: "Actual", mode: mode,
                marker: {size: pointSize, color: actualsColor}, line: {width: lineWidth, color: actualsColor},
            },
            {
                x: xAxis.slice(plotValues.length, xAxis.length), y: preds, name: "Forecast", mode: mode,
                marker: {size: pointSize, color: predsColor}, line: {width: lineWidth, color: predsColor}
            },
        ],
        {xaxis: {tickvals: xAxis, ticktext: tickText}, margin: {t: 0, r: 0}, legend: {orientation: "h", x: 0, y: 1}});

    removeLoadingMessage();
}


function removeOptions(selectElement) {
    var i, L = selectElement.options.length - 1;
    for (i = L; i >= 0; i--) {
        selectElement.remove(i);
    }
}


function resetOptionInputs() {
    let defaultOption = 'NA'
    removeOptions(labelColInput)
    labelColInput.add(new Option(defaultOption, defaultOption, true))

    removeOptions(dateColInput)
    dateColInput.add(new Option(defaultOption, defaultOption, true))
}

function emptyTable() {
    $(dataTableId + ' thead').empty()
    $(dataTableId + ' tbody').empty()
}


function parseComplete(results) {
    let colNames = results.meta.fields;
    let labelOptions = labelColInput.options;
    let dateOptions = dateColInput.options;

    let tableColumns = []
    colNames.forEach(function (col) {
        labelOptions.add(new Option(col, col, false))
        dateOptions.add(new Option(col, col, false))
        tableColumns.push({data: col})
    })
    if (colNames.length > 1) {
        $(dateColInput).val(colNames[0]);
        $(labelColInput).val(colNames[1]);
    } else {
        $(labelColInput).val(colNames[0]);
    }

    rawData = results.data;

    writeTable(dataTableId, rawData.slice(0, 20))
    activateTab('data');
    // console.log(results)
}


$('#file').change(function (e) {
    resetOptionInputs();
    emptyTable();
    rawData = [];
    let file = document.getElementById('file').files[0];
    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        dynamicTyping: true,
        complete: parseComplete,
    })
});

$('#use_example').click(function (e) {
    e.preventDefault();
    resetOptionInputs();
    emptyTable();
    rawData = [];
    let filePath = 'https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv'
    Papa.parse(filePath, {
        header: true,
        download: true,
        skipEmptyLines: true,
        dynamicTyping: true,
        complete: parseComplete,
    })
})

$("#paramForm").on('submit', function (e) {
    e.preventDefault();
    runAnalysis();
})
