// browserify for_bundle.js -o bundle.js

// https://stackoverflow.com/questions/49562978/how-to-use-npm-modules-in-browser-is-possible-to-use-them-even-in-local-pc
// https://www.npmjs.com/package/yahoo-finance
var yahooFinance = require('yahoo-finance');
global.window.yahooFinance = yahooFinance

/*
// EDITED IN BUNDLE
const PROXY_URL = 'https://cors-anywhere.herokuapp.com/';
exports.HISTORICAL_CRUMB_URL = PROXY_URL + 'finance.yahoo.com/quote/$SYMBOL/history';
exports.HISTORICAL_DOWNLOAD_URL = PROXY_URL + 'query1.finance.yahoo.com/v7/finance/download/$SYMBOL';
exports.SNAPSHOT_URL = PROXY_URL + 'query2.finance.yahoo.com/v10/finance/quoteSummary/$SYMBOL';
 */


// https://github.com/albertosantini/node-quadprog
var qp = require('quadprog');
global.window.qp = qp


// https://www.npmjs.com/package/compute-covariance
var cov = require( 'compute-covariance' );
global.window.cov = cov






// https://github.com/gadicc/node-yahoo-finance2
// import yahooFinance from 'yahoo-finance2';
// var yahooFinance = require('yahoo-finance2').default; // NOTE the .default
// global.window.yahooFinance = yahooFinance

