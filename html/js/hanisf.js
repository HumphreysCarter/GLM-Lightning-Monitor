"use strict";
/** @constructor */
var HAniSF = function() {
  var HAniS = this;
  var userWindow, canH, canW, imgHeight, imgWidth, canXScale, canYScale,
  controls, bottomControls, firstlast, first, last, loadMsg, loadMsgAuto,
  pointer, debug, popupWindow, debugWindow, debugText, buttcss, undoText,
  ptr, useDiv=false, divall, divcon, divcont, divconb, divcan, divcanStyle,
  divanim, divtop, prefHgt, prefWid, imgCan,
  ctx, ctx1, drwCan, ctxd, topCan, ctxtop,
  usingZip, zipFilename, zipOnly, zipFile, zipStatic,
  backFilesUrl, numFrames, backFiles, useWheelFrame, divoldrop,
  numOverlays, overlayTop, overlayLabels,overlayOrder, overlayFiles, overlayFilesUrl,
  overlayLinks, fofBase, imageBase, configImageBase, backImages, overlayImages,
  overlayCheck, overlayAlpha, overlayStatic, showTip, tipText, tipX, tipY,
  overlayClear, useForAll, overlaySlice, sliceCoords, sliceSmoothing,
  redirect, redirectList, xScreen, yScreen,
  xLoc, yLoc, xImage, yImage, wImage, hImage, xMove,yMove, drawPrompts, useCN,
  begFrame, isRunning, isLooping, wasLooping, refLooping,
  curFrame, cfChanged, direction, isRocking,
  fetchImages, needSizes=true, chkImageSize, isCtrlKey, noShortCuts,
  configValues, dwell, minDwell, maxDwell, dwChanged, cdLoading,
  stepDwell, lastDwell, delay, enableSmoothing,
  enhance, enhTab, isBaseEnh, isOverlayEnh, overlayEnhNum,
  enhCan, origCan, ctxe, ctxo, origIDd, enhID, autoEnhanceBg,
  aeCan, ctxae, ctxaed, eod, eodk, etr, etg, etb, eta, esd, ek, autoEnhanceList,
  toggleFrames, wTog, hTog, spTog, cwTog, divtog, togstart, togPointer,
  togColorOff, togColorOn, togColorSel, missTog, missTogColor,
  hotspots, numHotspots, isIconHotspot, backStatic,
  spriteImagesCnt, spriteImages, spriteImagesOffset, useSpriteFn,
  prevNumHotzones, numHotzones, hotzones, rgb, rgbpack,
  hoverzones, numHoverzones, doHoverzones, gotHoverzones, hoverCan, ctxh,
  hoverPick, showedHover, allowHoverzones, okToShowHoverzones,
  zoomFactors, zoomFactorIndex, hiResBase, hiResBaseName,
  hiResOlay, hiResOlayName, doingHiResZoom, frameLabels, frameLabelField,
  hiResOlayIndex, useToggle, toggleDefs, cantog, ctxtog, gotImages, doFOF,
  startstop, forward, backward, setSpeed, faster, slower, overlaySpacer,
  mark, isMark, markMode, markPrompts, markFont, markClose,
  markColor, markPoints, markSaveTB, markTB, markXbeg, markXend, markHbeg,
  extrap, isExtrap, extrapMode, extrapX, extrapY, extrapT, extrapXbeg,
  extrapPrompts, extrapYbeg, extrapYend, extrapYpos,
  extrapTB, extrapTimes, exsign, dxdt, dydt, dt,
  extrapTimesTemplate, minutes, doTime, xInc, yInc, tmin, nmin, exMsg,
  startingMinute, utcOffset, tzLabel, timeColor, timeBack, timeFont,
  timeFontSize, extrapAMPM, toFromLock, toFrom, showOpt,
  show, showPrompt, capture, capturePrompt, captureFilename,
  annotate, annotatePrompt, ants, annPointer,
  frameTimesRe, frameTimesDef, frameTimes, frameTimesFormat,
  saveall, saveallPrompt, saveallFilename, saveallList,
  divannot, isAnnot, anpe, anxdiff, anydiff, anxloc, anyloc, anyDown, anzoom,
  antype, ancol, anwid, antxt, anlastw, anlastf,
  zoom, keepZoom, isDown,
  zoomScale, zoomXFactor, zoomYFactor, zoomXBase, zoomYBase,
  isInitialZoom, initialZoom, initialX, initialY,
  wasZooming, enableZooming, zoomFactorMax, doZoomFactors, isDragging, cycleZoom,
  doTransparency=false, transRed, transGreen, transBlue,
  useTransparencyList, transparencyList, haltMe, wasHalted,
  looprock, loophalt, setframe, setframeLabel, setframeLabelSpan, isSetframe, distance,
  doDistance, doDirection, x0Dist, y0Dist, distDigits, distShift,
  xText, yText, wText, hText, winFirst=true, initCall, extraParams,
  x1Dist, y1Dist, distBox, distLineColor, markCursor, prevCursor, distCursor, locCursor,
  showDistance, distXScale, distYScale, distUnit, tipBox,
  location, doLocation, showLocation, locDigits, locLatPrefix, locLonPrefix,
  locBox, locTran, locll, llstr, loc0, loc1, loc2, loc3,
  preserveBackPoints, divolay, numdivolay, olayZoomIndex,
  preserveIndex, preservePoints, preserveAlways,
  restore, refresh, autotoggle, popupDiv, popupWinWidth, popupWinHeight,
  overlayProbe, showProbe, probe, tabGray, probeCursor,
  gotTable, tabUnit, tabPrefix, tabDecimal, minx, minDiff, probeBox,
  probeUndef, probeTest, probeExact, doBaseProbe,
  dirspdBox, dirspdPrefix, dirspdSuffix, dirspdLabel, dirspdX, dirspdY,
  diffInx, diffPct, pct, m1, m2, drgb, tn, diff, pValue, dbzz,
  tabVal, tabNum, tabDif, tabEnh, tabR, tabG, tabB, tabA, tabInx, tabMissing,
  lastFOF, isAutoRefresh, autoRefresh, refreshTimer, showProgress,
  spriteImg, spriteX, spriteY, spriteW, spriteH, gotSprites,
  useProgress, progX, progY,imgGotCount, imgCount, hideBottom, hideBackground,
  hideBottomDef, hideTop, hideTopDef, hideLeft, hideLeftDef,
  hideBottomZoom, hideTopZoom, hideRightZoom, hideLeftZoom,
  hideRight, hideRightDef, divalign, overlayBase;

  var compass = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW','N'];

  var requestAnimationFrame = window.requestAnimationFrame ||
     window.mozRequestAnimationFrame ||
     window.webkitRequestAnimationFrame ||
     window.msRequestAnimationFrame ||
     function(cb) {
       setTimeout(cb,10);
     };

  var make = function(t) {
    return document.createElement(t);
  };

  var beginsWith = function(strng, ltr) {
    if (strng == null) return;
    return (strng.trim().toLowerCase().indexOf(ltr) == 0);
  }

  var endsWith = function(strng, ltr) {
    if (strng == null) return;
    var trm = strng.trim();
    return (trm.lastIndexOf(ltr) === (trm.length-1))
  }

  var decodeHtml = function(shtml) {
    if (undoText == undefined) undoText = make("textarea");
    undoText.innerHTML = shtml;
    return undoText.value;
  }

  var makeSelect = function(lab,val, evt) {
    var s = make("select");
    s.style.width="70px";
    s.style.margin="5px";
    for (var i=0; i<lab.length;i++) {
      var opt = make("option");
      opt.innerHTML = lab[i];
      opt.value = val[i];
      s.add(opt);
    }
    s.addEventListener("change", evt, false);
    return s;
  }

  var doOverlayFilenames = function(cv) {
    var a = cv.split(",");
    overlayFiles = new Array(numFrames);
    for (var i=0; i<a.length; i++) {
      var b = a[i].split("&");
      numOverlays = a.length;
      for (var j=0; j<numFrames; j++) {
        if (i === 0) overlayFiles[j] = new Array(numOverlays);

        var jj = (b.length == 1 ? 0 : j)
        overlayFiles[j][i] = b[jj].trim();

        if (overlayBase != null) {
          overlayFiles[j][i] = (overlayBase+overlayFiles[j][i]).trim();
        } else if (imageBase != null) {
          overlayFiles[j][i] = (imageBase+overlayFiles[j][i]).trim();

        }

      }

    }

    doFOF = false;
  }

  var doFilenames = function(cv) {
    var bfn = cv.split(",");
    numFrames = bfn.length;
    backFiles = new Array(numFrames);
    for (var i=0; i<numFrames; i++) {
      if (imageBase != null) {
        backFiles[i] = imageBase+bfn[i].trim();
      } else {
        backFiles[i] = bfn[i].trim();
      }
    }

    doFOF = false;
    if (numFrames > numOverlays) numOverlays = 0;
  }

  var doBasename = function(cv, bsv, binc) {
    var bn = cv;
    if (imageBase != null) {
       bn = imageBase+cv;
    }
    backFiles = new Array(numFrames);
    var val;
    for (var i=0; i<numFrames; i++) {
      val = i*binc + bsv;
      if (bn.indexOf("*") >= 0) {
        backFiles[i] = bn.replace("*", val);

      } else if (bn.indexOf("?") >= 0) {
        var subbn = bn;
        while ( subbn.lastIndexOf("?") >= 0) {
          var li = subbn.lastIndexOf("?");
          var ts = subbn.substring(0,li)+(val % 10) +subbn.substring(li+1);
          subbn = ts;
          val = Math.floor(val / 10);
        }
        backFiles[i] = subbn;
      } else {
        backFiles[i] = bn + i;
      }

      info("image filename = "+backFiles[i]);
    }

    cv = configValues["reverse_order"];
    if (cv != null && beginsWith(cv,"t")) backFiles.reverse();

    doFOF = false;
    if (numFrames > numOverlays) numOverlays = 0;
  }

  var nosel = "position:relative;border:0px;margin:0px;padding:0px;-webkit-user-select:none; -moz-user-select:none; -ms-user-select:none; user-select:none; -webkit-tap-heightlight-color:transparent;";

  // polyfill for toBlob...
  if (!HTMLCanvasElement.prototype.toBlob) {
   Object.defineProperty(HTMLCanvasElement.prototype, 'toBlob', {
    value: function (callback, type, quality) {

      var binStr = window.atob( this.toDataURL(type, quality).split(',')[1] ),
          len = binStr.length,
          arr = new Uint8Array(len);

      for (var i=0; i<len; i++ ) {
       arr[i] = binStr.charCodeAt(i);
      }

      callback( new Blob( [arr], {type: type || 'image/png'} ) );
    }
   });
  }

  this.setup = function(confn, divnam, initCallback, extras) {

    if (divnam == undefined) divnam = "MORdivcan";
    initCall = initCallback;
    if (initCall != undefined && initCall.trim().length < 1 ) initCall = null;
    if (extras != undefined) {
      extraParams = extras.split("\n");
    } else {
      extraParams = null;
    }

    divall = document.getElementById(divnam);
    divalign = "center";
    divall.align=divalign;

    divcan = make("div");
    divcan.setAttribute("style", nosel);

    var canpos = "position:absolute;top:0;left:0;z-index:";
    imgCan = make("canvas");
    imgCan.setAttribute("style",canpos+"1;");
    ctx = imgCan.getContext("2d");

    drwCan = make("canvas");
    drwCan.setAttribute("style",canpos+"6;");
    ctxd = drwCan.getContext("2d");
    ctxd.imageSmoothingEnabled = false;

    topCan = make("canvas");
    topCan.setAttribute("style",canpos+"10;");
    ctxtop = topCan.getContext("2d");
    ctxtop.imageSmoothingEnabled = false;

    divanim = make("div");
    divanim.setAttribute("style",canpos+"5;overflow:hidden;border-width:0px;padding:0px;margin:0px;");

    divtop = make("div");
    divtop.setAttribute("style",canpos+"10;overflow:hidden;border-width:0px;padding:0px;margin:0px;");
    divtop.appendChild(topCan);

    aeCan = make("canvas");

    divcan.appendChild(divtop);
    divcan.appendChild(imgCan);
    divcan.appendChild(drwCan);
    divcan.appendChild(divanim);

    divall.appendChild(divcan);

    pointer = new PEvs(divcan, HAniS.down, HAniS.up,
         HAniS.move, HAniS.drag, null, HAniS.canTip);

    divall.setAttribute("tabindex",0);
    divall.addEventListener("keyup", function(e) {
      if (isAnnot || noShortCuts) return;
      isCtrlKey = false;
      e.preventDefault();
      e.stopPropagation();
    }, false);

    divall.addEventListener("keypress", function(e) {
      if (isAnnot || noShortCuts) return;
      e.preventDefault();
      e.stopPropagation();
    }, false);

    divall.addEventListener("keydown", function(e) {
      if (isAnnot || noShortCuts) return;
      e.preventDefault();
      e.stopPropagation();
      if (e.ctrlKey) {
        isCtrlKey = true;
      } else {
        isCtrlKey = false;
      }
      if (e.keyCode == 37) {  // arrowleft
          setIsLooping(false);
          incCurrentFrame(-1);
          drawIt();
      } else if (e.keyCode == 39) {  // arrowright
          setIsLooping(false);
          incCurrentFrame(+1);
          drawIt();
      } else if (e.keyCode == 32 ) {  // spacebar
          HAniS.toggleIsLooping();
      } else if (e.keyCode == 82 && e.altKey) {  //alt+r
          HAniS.resetZoom();
      }
    }, false);

    divtop.addEventListener("mouseenter", function(e) {
       divall.focus();
    }, false);

    divall.style.outlineWidth = "0px";
    divall.style.outline = "none";
    divall.focus();

    curFrame = 0;
    begFrame = 0;
    isRocking = false;
    direction = +1;
    haltMe = false;
    wasHalted = false;
    isLooping = true;
    wasLooping = true;
    numFrames = 0;
    numOverlays = 0;
    numHotspots = 0;
    chkImageSize = true;
    overlayOrder = null;
    fetchImages = true;
    dwell = 500;
    minDwell = 100;
    maxDwell = 2000;
    stepDwell = 30;
    lastDwell = 0;
    autoRefresh = 60000;
    refreshTimer = null;
    isAutoRefresh = false;
    usingZip = false;
    zipOnly = true;
    backFilesUrl = [];
    imageBase = null;
    overlayBase = null;
    fofBase = null;
    configImageBase = null;
    zoomFactors = null;
    doingHiResZoom = false;
    zoomFactorIndex = -1;
    doZoomFactors = false;
    zoomXFactor = 1.0;
    zoomYFactor = 1.0;
    zoomXBase = 1.0;
    zoomYBase = 1.0;
    isDragging = false;
    xScreen = 0;
    yScreen = 0;
    xLoc = 0;
    yLoc = 0;
    xImage = 0;
    yImage = 0;
    wImage = 0;
    hImage = 0;
    xMove = 0;
    yMove = 0;
    isInitialZoom = false;
    winFirst = true;
    needSizes = true;
    useDiv = false;
    isDown = false;
    loadMsgAuto = false;
    useToggle = false;
    missTog = -1;
    toggleFrames = new Array();
    numHotzones = 0;
    isIconHotspot=false;
    spriteImages = null;
    spriteImagesOffset = -1;
    spriteImagesCnt = 0;
    useSpriteFn = false;
    frameLabelField = null;
    imgCount = 0;
    imgGotCount = 0;
    progX = 0;
    progY = 0;
    tabEnh = false;
    tn = 0;
    gotImages = false;
    backStatic = false;
    zipStatic = false;
    showTip = false;
    x0Dist = y0Dist = x1Dist = y1Dist = 0;
    doDistance = false;
    showDistance = false;
    prevCursor = [];
    prevCursor.push("default");
    prevCursor.push("default");
    distCursor = "default";
    doLocation = false;
    showLocation = false;
    locCursor = "default";
    locTran = null;
    locll = [0,0];
    distShift = false;
    isSetframe = false;
    doHoverzones = false;
    okToShowHoverzones = true;
    allowHoverzones = null;
    hoverPick = null;
    showedHover = false;
    isMark = false;
    markClose = true;
    markCursor = "default";
    drawPrompts = true;
    markPoints = [];
    doBaseProbe = false;
    showProbe = false;
    probeCursor = "default";
    gotTable = false;
    m1 = [1,0,0];
    m2 = [2,2,1];
    drgb = [];
    overlayProbe = [];
    isExtrap = false;
    isBaseEnh = false;
    isOverlayEnh = false;
    enhTab = 0;
    exMsg = 0;
    utcOffset = 0;
    divtog = null;
    divcon = null;
    divcont = null;
    divconb = null;
    divolay = null;
    numdivolay = 0;
    isCtrlKey = false;
    noShortCuts = false;
    isAnnot = false;

    var can1 = make("canvas");
    can1.height=1;
    can1.width=1;
    ctx1 = can1.getContext("2d");
    ctx1.imageSmoothingEnabled=false;

    hoverCan = make("canvas");

    debug = false;
    if (typeof confn === "string") {
      if (confn.indexOf("\n") < 0) {
        getConfig(confn);
      } else {
        parseConfig(confn.split("\n"));
      }
    } else {
      parseConfig(confn);
    }

  }

  function parseConfig (txt) {
    var i,j, st, steq, sto, cv;
    doFOF = true;
    if (txt instanceof Array) {
      configValues = {};
      if (extraParams != null) txt = txt.concat(extraParams);
      for (i=0; i<txt.length; i++) {
        if (txt[i].length < 2) continue;
        st = txt[i].trim();
        if (st.indexOf("#") == 0) continue;

        steq = st.indexOf("=");
        if (steq < 1) continue;
        sto = st.substr(0,steq).trim().toLowerCase();
        configValues[sto] = st.substr(steq+1).trim();
      }
    } else {
      configValues = txt;
    }

    debug = false;
    cv = configValues["debug"];
    if (cv != null) {
      if (cv == "true") {
        debugWindow = window.open("","HAniSDebugInfo","scrollbars=yes,width=400,height=200");
        debug = true;
        info("HAniS Version 4.28");
      } else {
        debug = false;
      }
    }

    useCN = true;
    cv = configValues["set_className"];
    if (cv != null) {
      if (beginsWith(cv,"f")) useCN = false;
    }

    info("Is type of param an array = "+(txt instanceof Array));

    cv = configValues["divalign"];
    if (cv != null) {
      divalign = cv.trim();
      divall.align = divalign;
    }

    cv = configValues["prevent_shortcuts"];
    if (cv != null) {
      if (beginsWith(cv, "t")) noShortCuts = true;
    }

    cv = configValues["initial_message"];
    if (cv != null) {
      var lms = configValues["initial_message_style"];
      if (lms == null) {
        lms = "background-color:red;top:10px;left:10px;height:100px;width:200px";
      }
      lms = "z-index:99;position:absolute;"+lms;
      loadMsg = make("div");
      loadMsg.setAttribute("style",lms);
      loadMsg.innerHTML = cv;
      loadMsg.addEventListener("click", function() {
        loadMsg.style.visibility = "hidden";
      },false);
      loadMsg.style.visibility = "visible";
      divcan.appendChild(loadMsg);
      var lmst = configValues["initial_message_timeout"];
      if (lmst != null) {

        if (lmst == "auto") {
          loadMsgAuto = true;
        } else {
          loadMsgAuto = false;
          var imt = parseInt(configValues["initial_message_timeout"],10);
          if (imt < 2) imt = 2;
          setTimeout(function() {
            loadMsg.style.visibility = "hidden";
          }, imt*1000 );
        }
      }
    }


    cv = configValues["coordinates"];
    if (cv != null) {
      parseCoordinates(cv);
    }
    var locBGColor = "green";
    var locFGColor = "white";
    var locFont = "12px Arial";
    var locScolor = null;
    var locSblur = 10;
    var locSxoff = 10;
    var locSyoff = 10;
    locDigits = 2;
    locLatPrefix = "Lat = ";
    locLonPrefix = " Lon = ";

    cv = configValues["coordinates_display_style"];
    if (cv != null) {
      var a = cv.split(",");
      locBGColor = a[0].trim();
      locFGColor = a[1].trim();
      locFont = a[2].trim();
      if (a.length > 3) {
        locLatPrefix = a[3];
        locLonPrefix = a[4];
        locDigits = parseInt(a[5],10);
        if (a.length > 6) {
          locScolor = a[6].trim();
          locSblur = parseInt(a[7], 10);
          locSxoff = parseInt(a[8], 10);
          locSyoff = parseInt(a[9], 10);
        }
      }
    }
    locBox = new TextBox(locBGColor, locFGColor, locFont,
        locScolor, locSblur, locSxoff, locSyoff);

    distXScale = null;
    distYScale = null;
    if (configValues["map_scale"] != null) {
      doDistance = true;
      distShift = true;
      var a = configValues["map_scale"].split(",");
      distXScale = parseFloat(a[0]);
      if (a.length > 1) {
        distYScale = parseFloat(a[1]);
      } else {
        distYScale = distXScale;
      }
    }

    distUnit = " ";
    cv = configValues["distance_unit"];
    if (cv != null) {
      distUnit = cv.trim();
    }

    divcanStyle = " ";
    cv = configValues["divcan_style"];
    if (cv != null) {
      divcanStyle = cv.trim();
    }

    distCursor = "default";
    cv = configValues["distance_cursor"];
    if (cv != null) {
      distCursor = cv.trim();
    }

    locCursor = "default";
    cv = configValues["location_cursor"];
    if (cv != null) {
      locCursor = cv.trim();
    }

    probeCursor = "default";
    cv = configValues["probe_cursor"];
    if (cv != null) {
      probeCursor = cv.trim();
    }

    markCursor = "default";
    cv = configValues["mark_cursor"];
    if (cv != null) {
      markCursor = cv.trim();
    }

    cv = configValues["imagecan_style"];
    if (cv != null) {
      var a = cv.trim()+";position:absolute;top:0;left:0;";
      imgCan.setAttribute("style",a+"z-index:1;");
      drwCan.setAttribute("style",a+"z-index:6;");
    }

    var distBGColor = "blue";
    var distFGColor = "white";
    var distFont = "12px Arial";
    var distScolor = null;
    var distSblur = 10;
    var distSxoff = 10;
    var distSyoff = 10;
    distDigits = 1;
    distLineColor = "white";
    cv = configValues["distance_display_style"];
    if (cv != null) {
      var a = cv.split(",");
      distBGColor = a[0].trim();
      distFGColor = a[1].trim();
      distLineColor = (a.length >= 5) ? a[4].trim() : distFGColor;
      distFont = a[2].trim();
      distDigits = parseInt(a[3],10);
      if (a.length > 5) {
        distScolor = a[5].trim();
        distSblur = parseInt(a[6], 10);
        distSxoff = parseInt(a[7], 10);
        distSyoff = parseInt(a[8], 10);
      }
    }

    distBox = new TextBox(distBGColor, distFGColor, distFont,
        distScolor, distSblur, distSxoff, distSyoff);

    var probeBG= "black";
    var probeFG= "white";
    var probeFont = "12px Arial";
    var probeScolor = null;
    var probeSblur = 10;
    var probeSxoff = 10;
    var probeSyoff = 10;

    cv = configValues["probe_display_style"];
    if (cv != null) {
      var a = cv.split(",");
      probeBG = a[0].trim();
      probeFG = a[1].trim();
      probeFont = a[2].trim();
      if (a.length > 3) {
        probeScolor = a[3].trim();
        probeSblur = parseInt(a[4], 10);
        probeSxoff = parseInt(a[5], 10);
        probeSyoff = parseInt(a[6], 10);
      }
    }
    probeBox = new TextBox(probeBG, probeFG, probeFont,
        probeScolor, probeSblur, probeSxoff, probeSyoff);

    probeUndef = "Undefined";
    probeExact = false;
    probeTest = 0;

    cv = configValues["probe_undefined"];
    if (cv != null) {
      var a = cv.split(",");
      probeUndef = a[0].trim();
      if (a.length > 1) probeTest = parseInt(a[1], 10) + 1;
      if (a.length > 2) probeExact = true;
    }

    doBaseProbe = false;
    if (configValues["probe_base_image"]) doBaseProbe = true;

    cv = configValues["active_probe"];
    if (cv != null) {
      showProbe = true;
    }


    cv = configValues["pause"];
    if (cv != null) {
      lastDwell = parseInt(cv,10);
    }

    cv = configValues["auto_refresh"];
    if (cv != null) {
      autoRefresh= parseInt(cv,10)*60000;
      refreshTimer = setInterval("HAniS.reloadFOF();",autoRefresh);
      isAutoRefresh = true;
    }

    enableSmoothing = false;
    cv = configValues["enable_smoothing"];
    if (cv != null) {
      if (beginsWith(cv, "t")) enableSmoothing = true;
    }

    chkImageSize = true;
    cv = configValues["check_image_size"];
    if (cv != null) {
      if (beginsWith(cv, "f")) chkImageSize = false;
    }

    gotSprites = false;
    cv = configValues["sprites"];
    if (cv != null) {
      var a = cv.split(",");
      var spfn = a[0];
      spriteImg = new Image();
      spriteX = [];
      spriteY = [];
      spriteW = [];
      spriteH = [];

      if (a[1].trim().toLowerCase() == "row") {
        var xp=0, yp=0, wp, hp, np, mhp=0;
        i=2;
        while (i < a.length) {
          if (a[i].trim().toLowerCase() == "row") {
            xp = 0;
            yp = yp + mhp;
            mhp = 0;
            i=i+1;
          }
          np = parseInt(a[i],10);
          i=i+1;
          wp = parseInt(a[i],10);
          i=i+1;
          hp = parseInt(a[i],10);
          if (hp > mhp) mhp = hp;
          i=i+1;
          for (var k=0; k<np; k++) {
            spriteX.push(xp);
            spriteY.push(yp);
            spriteW.push(wp);
            spriteH.push(hp);
            xp = xp + wp;
          }
        }
      } else {

        var t, usePrev = false;
        i = 1;
        while (i < a.length) {
          spriteX.push(parseInt(a[i],10));
          i=i+1;
          spriteY.push(parseInt(a[i],10));
          i=i+1;
          if (i >= a.length) {
            t = 0;
            usePrev = true;
          } else {
            t = parseInt(a[i],10);
          }
          if (t < 0) {
            usePrev = true;
            spriteW.push(Math.abs(t));
            i =i+ 1;
            t = parseInt(a[i++],10);
            spriteH.push(Math.abs(t));
          } else if (usePrev) {
            spriteW.push(spriteW[spriteW.length-1]);
            spriteH.push(spriteH[spriteH.length-1]);
          } else {
            spriteW.push(parseInt(a[i],10));
            i=i+1;
            spriteH.push(parseInt(a[i],10));
            i=i+1;
          }
        }
      }

      spriteImg.onload = function() {
        gotSprites = true;
      }
      spriteImg.src = a[0];
    }

    cv = configValues["overlay_labels"];
    if (cv != null) {
      overlayLabels = cv.split(",");
    }

    autoEnhanceBg = null;
    cv = configValues["auto_enhance_background"];
    if (cv != null) {
      autoEnhanceBg = parseInt(cv,10)-1;
    }

    autoEnhanceList = null;
    cv = configValues["auto_enhance"];
    if (cv != null) {
      var cv2 = configValues["overlay_probe_table"];
      var a = cv.split(",");
      autoEnhanceList = new Array(a.length);
      for (i=0; i<a.length; i++) {
        if (a[i].trim().toLowerCase() == "x") {
          autoEnhanceList[i] = -1;
        } else {
          autoEnhanceList[i] = parseInt(a[i].trim(),10) - 1;
        }
        if (cv2 == null) overlayProbe[i] = autoEnhanceList[i];
      }
    }

    isOverlayEnh = false;
    overlayEnhNum = -1;
    cv = configValues["overlay_enhance"];
    if (cv != null) {
      isOverlayEnh = true;
      overlayEnhNum = parseInt(cv,10)-1;
    }

    allowHoverzones = null;
    okToShowHoverzones = true;
    cv = configValues["overlay_allow_hoverzones"];
    if (cv != null) {
      var a = cv.split(",");
      allowHoverzones = new Array(a.length);
      for (i=0; i<a.length; i++) {
        allowHoverzones[i] = true;
        if (beginsWith(a[i], "n") || beginsWith(a[i], "f"))   {
          allowHoverzones[i] = false;
        }
      }
    }

    canH = null;
    canW = null;
    userWindow = false;
    useDiv = false;
    cv = configValues["window_size"];
    if (cv != null) {
      var a = cv.split(",");
      if (a[0] == "div") {
        useDiv = true;
      } else {
        canW = parseInt(a[0], 10);
        canH = parseInt(a[1], 10);
      }
      userWindow = true;
    }

    useForAll = [];
    cv = configValues["use_for_all_frames"];
    if (cv != null) {
      var a = cv.split(",");
      for (i=0; i<a.length; i=i+2) {
        useForAll[parseInt(a[i],10)] = parseInt(a[i+1],10) - 1;
      }
    }

    missTog = -1;
    cv = configValues["skip_missing"];
    if (cv != null) {
      missTog = parseInt(cv, 10);
    }
    missTogColor = "#00000000";
    cv = configValues["skip_missing_color"];
    if (cv != null) {
      missTogColor = cv;
    }

    overlaySpacer = null;
    cv = configValues["overlay_spacer"];
    if (cv != null) {
      var a = cv.split(",");
      overlaySpacer = new Array(a.length);
      for (i=0; i<a.length; i++){
        overlaySpacer[i] = parseInt(a[i],10);
      }
    }

    cv = configValues["overlay_probe_table"];
    if (cv != null) {
      var a = cv.split(",");
      for (i=0; i<a.length; i++){
        if (a[i].trim().toLowerCase() == "x") {
          overlayProbe[i] = -1;
        } else {
          overlayProbe[i] = parseInt(a[i],10) - 1;
        }
      }
    }


    overlayTop = [];
    cv = configValues["overlay_ontop"];
    if (cv != null) {
      var a = cv.split(",");
      for (i=0; i<a.length; i++) {
        overlayTop[parseInt(a[i],10)-1] = true
      }
    }

    overlayOrder = null;
    cv = configValues["overlay_order"];
    if (cv == null) cv = configValues["overlay_zorder"];
    if (cv != null) {
      var a = cv.split(",");
      overlayOrder = new Array(a.length);
      for (i=0; i<a.length; i++){
        overlayOrder[parseInt(a[i],10) - 1] = i;
      }
    }

    overlayLinks = null;
    cv = configValues["overlay_links"];
    if (cv != null) {
      var a = cv.split(",");
      overlayLinks = new Array(a.length);
      for (i=0; i<a.length; i++) {
        overlayLinks[i] = parseInt(a[i],10);
      }
    }

    overlaySlice = [];
    sliceCoords = null;
    cv = configValues["overlay_slice"];
    if (cv != null) {
      var a = cv.split(",");
      for (i=0; i<a.length; i++) {
        overlaySlice[parseInt(a[i],10)-1] = true;
      }
    }

    sliceSmoothing = true;
    cv = configValues["slice_smoothing"];
    if (beginsWith(cv,"f")) sliceSmoothing = false;

    overlayClear = null; // 'n', 'z', 's', 'r'
    cv = configValues["overlay_clear"];
    if (cv != null) {
      var a = cv.split(",");
      overlayClear = new Array(a.length);
      for (i=0; i<a.length; i++) {
        overlayClear[i] = a[i].trim().toLowerCase();
      }
    }

    doTransparency = false;
    cv = configValues["transparency"];
    if (cv != null) {
      var a = cv.split(",");
      if (a.length == 1) {
        transRed = transGreen = transBlue = parseInt(a[0],10);
      } else {
        transRed = parseInt(a[0],10);
        transGreen = parseInt(a[1],10);
        transBlue = parseInt(a[2],10);
      }
      doTransparency = true;
    }

    useTransparencyList = false;
    cv = configValues["transparency_list"];
    if (cv != null) {
      var a = cv.split(",");
      transparencyList = new Array();
      for (i=0; i<a.length; i++) {
        transparencyList[parseInt(a[i].trim(),10)-1] = true;
      }
      useTransparencyList = true;
    }



    overlayAlpha = null;
    cv = configValues["overlay_transparent_amount"];
    if (cv != null) {
      var a = cv.split(",");
      overlayAlpha = new Array(a.length);
      for (i=0; i<a.length; i++){
        overlayAlpha[i] = parseFloat(a[i])/100.0;
      }
    }

    olayZoomIndex = null;
    cv = configValues["overlay_zoom"];
    if (overlayLabels != null && cv != null) {
      var a = cv.split(",");
      olayZoomIndex = new Array(a.length);
      for (i=0; i<a.length; i++) {
        olayZoomIndex[i] = 1;
        if (beginsWith(a[i],"n") || beginsWith(a[i], "f")) {
          olayZoomIndex[i] = 0;
        } else if (beginsWith(a[i],"d") || beginsWith(a[i], "h")) {
          olayZoomIndex[i] = -1;
        }
      }
    }

    preserveIndex = null;
    preserveAlways = [];
    cv = configValues["overlay_preserve_list"];
    if (overlayLabels != null && cv != null) {
      var a = cv.split(",");
      preserveIndex = new Array(a.length);
      for (i=0; i<a.length; i++) {
        preserveIndex[i] = true;
        if (beginsWith(a[i],"n") || beginsWith(a[i],"f")) {
          preserveIndex[i] = false;
        }
        if (beginsWith(a[i],"a") || beginsWith(a[i],"A")) {
          preserveAlways[i] = true;
        }
      }
    }

    preservePoints = null;
    if (overlayLabels != null && preserveIndex != null) {
      // syntax: = 0,0,10,10 & 100,100,150,150 , 80,80,100,100 ...
      var a = configValues["overlay_preserve"].replace(/&/g, ",&,").split(",");
      var b = configValues["overlay_preserve_position"];
      if (b != null) {
         b = b.split(",");
      }
      var bc = 0;
      preservePoints = new Array(preserveIndex.length);
      var p = 0;
      var again;
      for (i=0; i<preserveIndex.length; i++) {
        if (preserveIndex[i]) {
          preservePoints[i] = [];
          var kp = 0;
          do {
            preservePoints[i][kp] = [];
            for (j=0; j<4; j++) {
              preservePoints[i][kp][j] = parseInt(a[p].trim(), 10);
              p = p + 1;
            }
            // convert to width and height...
            preservePoints[i][kp][2] = preservePoints[i][kp][2] - preservePoints[i][kp][0] + 1;
            preservePoints[i][kp][3] = preservePoints[i][kp][3] - preservePoints[i][kp][1] + 1;
            if (b != null) {
              preservePoints[i][kp][4] = parseInt(b[bc],10);
              preservePoints[i][kp][5] = parseInt(b[bc+1],10);
              bc = bc + 2;
            } else {
              preservePoints[i][kp][4] = preservePoints[i][kp][0];
              preservePoints[i][kp][5] = preservePoints[i][kp][1];
            }

            if (p < a.length && a[p].trim() == "&") {
              p = p + 1;
              kp = kp + 1;
              again = true;
            } else {
              again = false;
            }
          } while (again);

        }
      }
    }

    utcOffset = 0;
    tzLabel = " ";
    timeColor = "white";
    timeBack = "transparent";
    timeFont = "14pt arial";
    extrapAMPM = false;
    // = color, utc_offset, tzoneLabel, backgnd, font, AMPM (true/false)
    cv = configValues["times_label_style"];
    if (cv != null) {
      var a = cv.split(",");
      timeColor = a[0].trim();
      if (a.length > 1)
      utcOffset = Math.floor(60. * parseFloat(a[1]));
      if (a.length > 1) {
        if (a[1].trim() == "?" || a[1].trim().toLowerCase() == "auto") {
          var dt = new Date();
          utcOffset = -Math.floor(dt.getTimezoneOffset());
        } else {
          utcOffset = Math.floor(60.  *parseFloat(a[1]));
        }
      }
      if (a.length > 2) tzLabel = a[2].trim();
      if (a.length > 3) timeBack = a[3].trim();
      if (a.length > 4) timeFont = a[4].trim();
      if (a.length > 5 && a[5].trim().indexOf("t")!=-1) extrapAMPM = true;
    }
    if (timeColor.indexOf("0x") == 0) {
      timeColor = "#"+timeColor.substring(2);
    }
    if (timeBack.indexOf("0x") == 0) {
      timeBack = "#"+timeBack.substring(2);
    }
    timeFontSize = parseInt(timeFont,10);
    extrapTB = new TextBox("black","white",timeFont, "black",20,5,5);

    toFromLock = false;
    toFrom = false;
    cv = configValues["to_from_lock"];
    if (cv != null) {
      if (beginsWith(cv,"t")) {
        toFrom = true;
      } else {
        toFrom = false;
      }
    }

    var dsboxBG = "black";
    var dsboxFG = "white";
    var dsboxFont = "14px arial";
    var dsboxScolor = null;
    var dsboxSblur = 10;
    var dsboxSxoff = 10;
    var dsboxSyoff = 10;
    dirspdSuffix = " ";
    dirspdPrefix = " ";
    dirspdLabel = null;
    dirspdBox = null;

    cv = configValues["dirspd_display_style"];
    if (cv != null) {
      var a = cv.split(",");
      // bg, fg, font, prefix, suffix, shadow color, blur, xoff,  yoff
      dsboxBG = a[0].trim();
      dsboxFG = a[1].trim();
      dsboxFont = a[2].trim();
      dirspdPrefix = a[3];
      dirspdSuffix = a[4];
      if (a.length > 5) {
        dsboxScolor = a[5].trim();
        dsboxSblur = parseInt(a[6], 10);
        dsboxSxoff = parseInt(a[7], 10);
        dsboxSyoff = parseInt(a[8], 10);
      }

      dirspdBox = new TextBox(dsboxBG, dsboxFG, dsboxFont,
          dsboxScolor, dsboxSblur, dsboxSxoff, dsboxSyoff);
    }

    var tipboxBG = "black";
    var tipboxFG = "white";
    var tipboxFont = "14px arial";
    var tipboxScolor = null;
    var tipboxSblur = 10;
    var tipboxSxoff = 10;
    var tipboxSyoff = 10;

    cv = configValues["tipbox_display_style"];
    if (cv != null) {
      var a = cv.split(",");
      // bg, fg, font, shadow color, blur, xoff,  yoff
      tipboxBG = a[0].trim();
      tipboxFG = a[1].trim();
      tipboxFont = a[2].trim();
      if (a.length > 3) {
        tipboxScolor = a[3].trim();
        tipboxSblur = parseInt(a[4], 10);
        tipboxSxoff = parseInt(a[5], 10);
        tipboxSyoff = parseInt(a[6], 10);
      }
    }

    tipBox = new TextBox(tipboxBG, tipboxFG, tipboxFont,
        tipboxScolor, tipboxSblur, tipboxSxoff, tipboxSyoff);


    fofBase = null;
    imageBase = configValues["image_base"];
    if (imageBase != null) {
      imageBase = imageBase.trim();
      fofBase = imageBase;
    } else if (configValues["image_only_base"] != null) {
      imageBase = configValues["image_only_base"].trim();
    } else {
      imageBase=null;
    }
    configImageBase = imageBase;

    cv = configValues["overlay_base"];
    if (cv != null) {
      overlayBase = cv.trim();
    }

    zipOnly = true;
    cv = configValues["zip_only"];
    if (cv != null) {
      if (beginsWith(cv, "f")) zipOnly = false;
    }

    usingZip = false;
    cv = configValues["zip_filename"];
    if (cv != null) {
      usingZip = true;
      zipFilename = cv.trim();
    }

    doFOF = true;
    cv = configValues["filenames"];
    if (cv == null) cv = configValues["background_filenames"];
    if (cv != null) {
      doFilenames(cv);
    }

    cv = configValues["basename"];
    if (cv != null) {
      // basename and get num_frames & compute filenames
      // allow for base_starting_number and increment
      // also, wildcards!!  (* means all digits, ??? means 3 digits)
      var b = configValues["base_starting_number"];
      var bsv = 0
      var binc = 1;
      if (b != null) {
        var a = b.split(",");
        bsv = parseInt(a[0],10);
        if (a.length > 1) {
          binc = parseInt(a[1],10);
        }
      }
      numFrames = configValues["num_frames"];
      if (numFrames == null) {
        numFrames = configValues["num_images"];
      }
      numFrames = parseInt(numFrames,10);
      doBasename(cv, bsv, binc);
    }

    zipStatic = true;
    var zst = configValues["zip_static"];
    if (zst != null) {
      if (beginsWith(zst,"f") || beginsWith(zst,"n")) {
        zipStatic = false;
      }
    }

    backStatic = true;
    var bst = configValues["background_static"];
    if (bst != null) {
      if (beginsWith(bst,"f") || beginsWith(bst,"n")) {
        backStatic = false;
      }
    }

    cv = configValues["overlay_filenames"];
    if (cv != null) {
      // overlay_filenames = oA1 & oA2 & oA3, oB1 & oB2 & oB3....
      doOverlayFilenames(cv);
    }

    preserveBackPoints = null;
    cv = configValues["image_preserve"];
    if (cv != null) {
      var a = cv.split(",");
      preserveBackPoints = new Array();
        for (i=0; i<a.length; i++) {
            preserveBackPoints[i] = parseInt(a[i].trim(), 10);
            // convert to width and height...
            if (i % 4 > 1) {
              preserveBackPoints[i] =
                 preserveBackPoints[i] - preserveBackPoints[i-2] + 1;
            }
        }
    }


    keepZoom = true;
    cv = configValues["keep_zoom"];
    if (cv != null) {
      if (beginsWith(cv,"f")) {
        keepZoom = false;
      }
    }

    hideBottomDef = 0;
    hideBottom = 0;
    hideBottomZoom = false;
    cv = configValues["hide_bottom"];
    if (cv != null) {
      var a = cv.split(",");
      hideBottomDef = parseInt(a[0].trim(),10);
      if (a.length > 1 && beginsWith(a[1],"t")) hideBottomZoom = true;
    }

    hideTopDef = 0;
    hideTop = 0;
    hideTopZoom = false;
    cv = configValues["hide_top"];
    if (cv != null) {
      var a = cv.split(",");
      hideTopDef = parseInt(a[0].trim(),10);
      if (a.length > 1 && beginsWith(a[1],"t")) hideTopZoom = true;
    }

    hideLeftDef = 0;
    hideLeft = 0;
    hideLeftZoom = false;
    cv = configValues["hide_left"];
    if (cv != null) {
      var a = cv.split(",");
      hideLeftDef = parseInt(a[0].trim(),10);
      if (a.length > 1 && beginsWith(a[1],"t")) hideLeftZoom = true;
    }

    hideRightDef = 0;
    hideRight = 0;
    hideRightZoom = false;
    cv = configValues["hide_right"];
    if (cv != null) {
      var a = cv.split(",");
      hideRightDef = parseInt(a[0].trim(),10);
      if (a.length > 1 && beginsWith(a[1],"t")) hideRightZoom = true;
    }

    hideBackground = false;
    cv = configValues["hide_background"];
    if (cv != null) {
      if (beginsWith(cv, "t")) hideBackground = true;
    }

    popupDiv = "<div>";
    popupWinHeight = 300;
    popupWinWidth = 500;
    cv = configValues["popup_style"];
    if (cv != null) {
      popupDiv = '<div style="'+cv+'">';
    }

    cv = configValues["popup_window_size"];
    if (cv != null) {
      var a = cv.split(",");
      popupWinWidth = a[0];
      popupWinHeight = a[1];
    }

    useWheelFrame = false;
    cv = configValues["wheel_frames"];
    if (cv != null) {
      useWheelFrame = true;
      pointer.useWheel(HAniS.wheel);
    }

    enableZooming = false;
    wasZooming = false;
    cv = configValues["active_zoom"];
    if (cv != null) {
      if (beginsWith(cv, "t")) {
        if (zoom == null) enableZooming = true;
        pointer.useWheel(HAniS.wheel);
      }
    }


    cycleZoom = false;
    cv = configValues["cycle_zoom"];
    if (cv != null) {
      if (beginsWith(cv, "t")) {
        cycleZoom = true;
      }
    }

    cv = configValues["initial_zoom"];
    if (cv != null) {
      var a = cv.split(",");
      if (a[0].toLowerCase() != "auto") {
        initialZoom = parseFloat(a[0]);
      } else {
        initialZoom = "auto";
      }
      if (a.length > 1) {
        initialX = parseInt(a[1],10);
        initialY = parseInt(a[2],10);
      } else {
        initialX = -1;
      }
      isInitialZoom = true;
    }

    zoomScale = 1.0;
    doZoomFactors = false;
    zoomFactors = null;
    cv = configValues["zoom_scale"];
    if (cv != null) {
      a = cv.split(",");
      if (a.length === 1) {
        if (beginsWith(cv, "hi_res")) {
          doZoomFactors = true;
          zoomScale = 0;
        } else  {
          zoomScale = parseFloat(cv);
        }
      } else {
        zoomFactors = new Array(a.length);
        for (i=0; i<a.length; i++) {
          zoomFactors[i] = parseFloat(a[i]);
        }
        doZoomFactors = true;
      }
    }

    zoomFactorMax = 9999999.0;
    cv = configValues["maximum_zoom"];
    if (cv != null) {
      zoomFactorMax = parseFloat(cv);
    }

    showProgress = true;
    cv = configValues["use_progress_bar"];
    if (cv != null) {
      if (beginsWith(cv,"f")) {
        showProgress = false;
      }
    }

    cv = configValues["dwell"];
    if (cv != null) {
      var a = cv.split(",");
      dwell = parseInt(a[0].trim(), 10);
      if (a.length > 1) {
        minDwell = parseInt(a[1].trim(), 10);
        if (minDwell < 10) minDwell = 10;
        maxDwell = parseInt(a[2].trim(), 10);
        stepDwell = parseInt(a[3].trim(), 10);
      }

    }
    cv = configValues["rate"];
    if (cv != null) {
      var a = cv.split(",");
      dwell = Math.round(10000./parseFloat(a[0].trim()));
      if (a.length > 1) {
        minDwell = Math.round(10000./parseFloat(a[2].trim()));
        if (minDwell < 10) minDwell = 10;
        maxDwell = Math.round(10000./parseFloat(a[1].trim()));
      }
    }

    frameTimesRe = null;
    frameTimesDef = null;
    cv = configValues["frame_times_template"];
    if (cv != null) {
      var a = cv.split(",");  // regexp, YYYY,MM,DD,hh,mm,ss
      frameTimesRe = RegExp(a[0]);
      frameTimesDef = a[1];
      frameTimes = [];
      frameTimesFormat = configValues["frame_times_format"];
    }

    extrapTimes = null;
    cv = configValues["times"];
    if (cv != null) {
      var a = cv.split(",");
      extrapTimes = new Array(a.length);
      for (i=0; i<a.length; i++) {
        extrapTimes[i] = parseInt(a[i].trim(), 10);
      }
      makeTimes(extrapTimes);
    }

    cv = configValues["extrap_times_filename"];
    if (cv != null ) {
      extrapTimes = new Array();
      var reqet = new XMLHttpRequest();
      reqet.open("get",cv,true);
      reqet.onload = function() {
        var txt = reqet.responseText.split("\n");
        for (var it=0; it<txt.length; it++) {
          var rt = txt[it].split(" ");
          if (rt.length == 3) {
            var rtim = rt[2].trim();
            var rti = parseInt(rt[0],10)-1;
            extrapTimes[rti] = parseInt(rt[2].substring(0,2),10)*100 + parseInt(rt[2].substring(3,5),10) + (parseInt(rt[2].substring(6,8),10)/30);
          }
        }
        makeTimes(extrapTimes);
      }
      reqet.send();

    }

    extrapTimesTemplate = null;
    cv = configValues["extrap_times_template"];
    if (cv != null) {
      extrapTimesTemplate = RegExp(cv);
    }

    extrapPrompts = ["Click on target's initial position","Click on target's final position","Move pointer around or click here to select target"];

    extrapYpos = 10;
    cv = configValues["extrap_prompts_position"];
    if (cv != null) {
      if (beginsWith(cv,"bot")) extrapYpos = 9999;
    }
    cv = configValues["extrap_prompts"];
    if (cv != null) {
      extrapPrompts = cv.split(",");
    }

    markPrompts = ["Click on first point to mark","Save points","Remove last point"];
    cv = configValues["mark_prompts"];
    if (cv != null) {
      markPrompts = cv.split(",");
    }

    markClose = true;
    cv = configValues["mark_close"];
    if (cv != null) {
      if (beginsWith(cv,"f")) markClose = false;
    }

    markFont = "12pt Arial";
    markColor = "white";
    cv = configValues["mark_prompts_style"];
    if (cv != null) {
      var mks = cv.split(",");
      markFont = mks[0].trim();
      if (mks.length > 1) markColor = mks[1].trim();
    }

    controls = configValues["controls"];
    info("controls = "+configValues["controls"]);
    info("bottom_controls = "+configValues["bottom_controls"]);

    startstop = null;
    firstlast = null;
    forward = null;
    backward = null;
    faster = null;
    slower = null;
    refresh = null;
    autotoggle = null;

    if (controls != null) {
      doMakeControls(controls, true);
    }

    bottomControls = configValues["bottom_controls"];
    if (bottomControls != null) {
      doMakeControls(bottomControls, false);
    }

    gotTable = false;
    cv = configValues["probe_table"];
    if (cv != null) {
      getProbeTable(cv);
    }

    // now do enh table, before loading images...
    var enhfile = configValues["enhance_filename"];
    if (enhfile == null) enhfile = configValues["enhance_table"];
    if (enhfile != null) {
      info("Reading enhancement table...");
      gotTable = false;

      var reqet = new XMLHttpRequest();
      tabVal = new Array();
      tabUnit = new Array();
      tabMissing = new Array();
      tabPrefix = new Array();
      tabDecimal = new Array();
      var tabPrevUnit, tabPrevPrefix, tabPrevDecimal;
      tabR = new Array();
      tabG = new Array();
      tabB = new Array();
      tabA = new Array();
      tabNum = -1;
      tabEnh = true;
      var gotValues = false;

      var inlo, inhi, indif, rlo, rhi, glo, ghi, blo, bhi, vlo, vhi, alo, ahi;

      reqet.onload = function() {
        var txt = this.responseText.split("\n");
        for (var i=0; i<txt.length; i++) {
          if (txt[i].length < 2) continue;

          var sr = txt[i].replace(/\s+/g,' ').trim();
          var m = sr.indexOf("#");
          if (m == 0) continue;
          if (m > 0) sr = sr.substring(0,m);
          var st = sr.split(",");

          if (st[0].indexOf("*") == 0) {
            tabNum++;
            tabVal[tabNum] = new Array(256);
            tabPrefix[tabNum] = new Array(256);
            tabUnit[tabNum] = new Array(256);
            tabDecimal[tabNum] = new Array(256);
            tabR[tabNum] = new Array(256);
            tabG[tabNum] = new Array(256);
            tabB[tabNum] = new Array(256);
            tabA[tabNum] = new Array(256);
            tabVal[tabNum] = new Array(256);
            if (st.length > 1) {
              tabMissing[tabNum] = st[1].trim();
            } else {
              tabMissing[tabNum] = "None";
            }
            tabPrevUnit = " ";
            tabPrevPrefix = "Value =";
            tabPrevDecimal = 1;

            var opt = make("option");
            opt.innerHTML = st[0].substring(1);
            opt.value = tabNum;
            if (enhance != null) enhance.add(opt);

          } else {

            var valInx = sr.indexOf("value");
            if (valInx < 0) valInx = sr.indexOf("probe");
            gotValues = false;
            vlo = vhi = 0;
            if (valInx > 0) {
              var valString = sr.substring(valInx+1);
              sr = sr.substring(0,valInx);
              var valEq = valString.indexOf("=");
              valString = valString.substring(valEq+1);
              var valItems = valString.split(",");
              if (valItems.length == 1) {
                tabPrevPrefix = valItems[0].trim();
                tabPrevDecimal = -1;
                gotValues = false;
              } else {
                vlo = parseFloat(valItems[0].trim());
                vhi = parseFloat(valItems[1].trim());
                if (valItems.length > 2) {
                  tabPrevDecimal = parseInt(valItems[2],10);
                }
                if (valItems.length > 3) {
                  tabPrevUnit = decodeHtml(valItems[3].trim());
                }
                if (valItems.length > 4) {
                  tabPrevPrefix = valItems[4].trim();
                }

                gotValues = true;
              }
            }

            var sp = sr.split(" ");
            inlo = parseInt(sp[0].trim(), 10);
            inhi = parseInt(sp[1].trim(), 10);
            indif = inhi - inlo;
            rlo = parseFloat(sp[2].trim());
            rhi = parseFloat(sp[3].trim());
            glo = parseFloat(sp[4].trim());
            ghi = parseFloat(sp[5].trim());
            blo = parseFloat(sp[6].trim());
            bhi = parseFloat(sp[7].trim());
            alo = ahi = 255;
            if (sp.length > 9) {
              alo = parseFloat(sp[8].trim());
              ahi = parseFloat(sp[9].trim());
            }

            for (var k=inlo; k<=inhi; k++) {
              if (indif === 0) {
                tabR[tabNum][inlo] = rlo;
                tabG[tabNum][inlo] = glo;
                tabB[tabNum][inlo] = blo;
                tabA[tabNum][inlo] = alo;
              } else {
                tabR[tabNum][k] = Math.round(rlo + (rhi - rlo) * (k - inlo) / indif);
                tabG[tabNum][k] = Math.round(glo + (ghi - glo) * (k - inlo) / indif);
                tabB[tabNum][k] = Math.round(blo + (bhi - blo) * (k - inlo) / indif);
                tabA[tabNum][k] = Math.round(alo + (ahi - alo) * (k - inlo) / indif);
              }
              tabPrefix[tabNum][k] = tabPrevPrefix;
              tabUnit[tabNum][k] = tabPrevUnit;
              tabDecimal[tabNum][k] = tabPrevDecimal;
              if (gotValues) {
                tabVal[tabNum][k] = vlo + (vhi - vlo) * (k - inlo) / indif;
              }
            }
          }
        }

        if ((overlayEnhNum != -1) && overlayCheck[overlayEnhNum].checked) {
          enhance.disabled = false;
        }

        gotTable = true;
        next2();
      }

      reqet.open("get",enhfile,true);
      reqet.send();

    } else {
      next2();
    }
  }

  function next2() {
    var cv = configValues["sprite_filenames"];
    if (cv != null) {
      var spritefn = cv.split(",");
      spriteImagesCnt = 0;
      spriteImages = new Array();
      var a = configValues["sprite_images_offset"];
      spriteImagesOffset = -1;
      if (a != null) spriteImagesOffset = parseInt(a,10) - 1;
      for (var k=0; k<spritefn.length; k++) {
        spriteImages[k] = new Image();
        spriteImages[k].gotit = false;
        spriteImages[k].inx = k;
        spriteImages[k].fromfile = true;
        var sep = "/";
        if (spritefn[k].indexOf("?") > 0) sep = "?";
        spriteImages[k].zoomme = (spritefn[k].indexOf(sep+"zoom") > 0);
        spriteImages[k].anim = (spritefn[k].indexOf(sep+"animate") > 0);
        var ss = spritefn[k].indexOf(sep);
        if (ss > 0) spritefn[k] = spritefn[k].substring(0,ss);
        spriteImages[k].onload = function() {
          spriteImages[this.inx].gotit = true;
          spriteImagesCnt++;
          if (spriteImagesCnt == spritefn.length) next3();
        }
        spriteImages[k].src = spritefn[k].trim();
      }

    } else {
      next3();
    }
  }

  function next3() {
    var hsk = Object.keys(configValues);
    for (var k=0; k<hsk.length; k++) {
      if (beginsWith(hsk[k],"hotspot")) {
        fetchHotspot("hotspot="+configValues[hsk[k]]);
      }
    }

    if (doFOF) {
      HAniS.getFOF();
    } else {
      fetchImages = true;
      loadImages();
    }
  }

  function getProbeTable(fn) {
    var reqpt = new XMLHttpRequest();
    tabVal = new Array();
    tabMissing = new Array();
    tabUnit = new Array();
    var tabPrevUnit = " ";
    tabPrefix = new Array();
    var tabPrevPrefix = "Value=";
    tabDecimal = new Array();
    var tabPrevDecimal = 1;
    tabDif = new Array(); // max diff in color 'tabInx', max of all
    tabInx = new Array();  // index (0=R, 1=G, 2=B) of max
    tabR = new Array(); // Red
    tabG = new Array(); // Green
    tabB = new Array(); // Blue
    var inx;
    var tabEnt = -1;
    tabNum = -1;
    var state = 0;

    reqpt.onload = function() {
      var txt = this.responseText.split("\n");
      for (var i=0; i<txt.length; i++) {
        if (txt[i].length < 2) continue;

        var sr = txt[i].replace(/\s+/g, ' ').trim();
        var st = sr.split(",");

        var sp = st[0].split(" ");

        if (sp[0].indexOf("*") == 0) {
          tabNum++;
          tabVal[tabNum] = new Array();
          tabPrefix[tabNum] = new Array();
          tabUnit[tabNum] = new Array();
          tabDecimal[tabNum] = new Array();
          tabR[tabNum] = new Array();
          tabG[tabNum] = new Array();
          tabB[tabNum] = new Array();
          tabInx[tabNum] = new Array();
          tabDif[tabNum] = new Array();
          tabEnt = -1;
          if (st.length > 1) {
            tabMissing[tabNum] = st[1].trim();
          } else {
            tabMissing[tabNum] = "None";
          }
          state = 1;
          continue;
        }
        if (state == 0) continue;

        tabEnt++;
        var dinx = new Array(3);
        tabVal[tabNum][tabEnt] = parseFloat(sp[0]);
        tabR[tabNum][tabEnt] = parseInt(sp[1].trim(),10);
        tabG[tabNum][tabEnt] = parseInt(sp[2].trim(),10);
        tabB[tabNum][tabEnt] = parseInt(sp[3].trim(),10);
        if (tabEnt != 0) {
          dinx[0] = tabR[tabNum][tabEnt] - tabR[tabNum][tabEnt-1];
          dinx[1] = tabG[tabNum][tabEnt] - tabG[tabNum][tabEnt-1];
          dinx[2] = tabB[tabNum][tabEnt] - tabB[tabNum][tabEnt-1];
          inx = 0;
          if (Math.abs(dinx[1]) > Math.abs(dinx[inx]) ) inx = 1;
          if (Math.abs(dinx[2]) > Math.abs(dinx[inx]) ) inx = 2;
          tabInx[tabNum][tabEnt-1] = inx;
          tabDif[tabNum][tabEnt-1] = dinx;
          if (tabDif[tabNum][tabEnt-1][inx] == 0) tabDif[tabNum][tabEnt-1][inx] = 1;
        }

        if (st.length > 1) {
          if (st[1].trim().indexOf('"') == 0) {
            tabPrevDecimal = -1;
            var ftb=st[1].trim();
            tabPrevPrefix = ftb.substr(1, ftb.length-2);
            tabPrevUnit = "";
          } else {
            tabPrevDecimal = parseInt(st[1],10);
            if (st.length > 2) tabPrevUnit = decodeHtml(st[2].trim());
            if (st.length > 3) tabPrevPrefix = st[3].trim();
          }
        }
        tabPrefix[tabNum][tabEnt] = tabPrevPrefix;
        tabDecimal[tabNum][tabEnt] = tabPrevDecimal;
        tabUnit[tabNum][tabEnt] = tabPrevUnit;
      }

      gotTable = true;

    }

    reqpt.open("get",fn,true);
    reqpt.send();

  }

  function parseCoordinates(st) {

    var a = st.split(",");
    var atm = a[0].trim();
    if (atm === "PS") {
      locTran = new MORhanPolarStereoEllips();
      locTran.init(a);

    } else if (atm == "CE") {
      locTran = new MORhanCylEqualDist();
      locTran.init(a);

    } else if (atm == "LCC") {
      locTran = new MORhanLambConConEllips();
      locTran.init(a);

    } else {
      locTran = null;
      loc0 = parseFloat(a[0]);
      loc1 = parseFloat(a[1]);
      loc2 = parseFloat(a[2]);
      loc3 = parseFloat(a[3]);
    }
  }

  function getConfig(fn) {
    var req = new XMLHttpRequest();
    req.onload = function() {
      parseConfig(this.responseText.split("\n"));
    }

    req.open("get", fn+"?"+Math.round(Math.random()*100000), true);
    req.send();

  }

  function clearOverlays(typ) {
    var i;
    if (overlayClear != null) {
      for (i=0; i<overlayClear.length; i++) {
        if (overlayClear[i].indexOf(typ) != -1) {
          overlayCheck[i].checked = false;
        }
      }
      for (i=0; i<overlayClear.length; i++) {
        resetLinks(i);
      }

    }
  }

  function makeTimes(s) {
    var ds = 0;
    minutes = new Array(s.length);
    for (var i=0; i<s.length; i++) {
      var mm = 60*Math.floor(s[i]/100) + Math.floor(s[i]) % 100;
      if (i === 0) startingMinute = mm;
      ds = mm - startingMinute;
      if (ds < 0) {  // test if gone over 00Z -- only allowed once!
        ds = mm + 24*60 - startingMinute;
      }
      minutes[i] = ds;
      info("times s="+mm+"  min="+minutes[i]);
    }
    startingMinute = startingMinute + utcOffset;
    if (startingMinute >= 1440) startingMinute = startingMinute - 1440;
    if (startingMinute < 0) startingMinute = startingMinute + 1440;
    info("Starting minute = "+startingMinute);
  }

  function setHotzoneVis(nz, vis) {
    if (nz != 0) {
      for (var i=0; i<nz; i++) {
        if (overlayCheck[hotzones[i].overlay].isGhost == false ) {
          overlayCheck[hotzones[i].overlay].parentNode.style.visibility=vis;
        }
      }
    }
  }

  this.newBasenames = function(basename, starting, increm, numf) {
    gotImages = false;
    wasLooping = isLooping;
    setIsLooping(false);
    numFrames = numf;
    doBasename(basename, starting, increm);
    fetchImages = true;
    loadImages();
  }

  this.newFilenames = function(names) {
    gotImages = false;
    wasLooping = isLooping;
    setIsLooping(false);
    doFilenames(names);
    fetchImages = true;
    loadImages();
  }

  this.newOverlayFilenames = function(names) {
    gotImages = false;
    wasLooping = isLooping;
    setIsLooping(false);
    doOverlayFilenames(names);
    fetchImages = true;
    loadImages();
  }

  this.newProbeTable = function(tbl) {
    gotTable = false;
    getProbeTable(tbl.trim());
  }

  this.reloadFOF = function() {
    info("reload FoF = "+refreshTimer);
    clearOverlays("r");
    if (doFOF) {
      processFOF(lastFOF, false);
    } else {
      if (refLooping == null) {
        wasLooping = true;  // old behavior
      } else {
        wasLooping = isLooping;
      }
      fetchImages = true;
      loadImages();
    }
  }

  this.getFOF = function() {
    var fn = configValues["file_of_filenames"].trim();
    var fofn =(fofBase != null ? fofBase+fn : fn);
    HAniS.newFOF(fofn, true);
  }

  this.newFOF = function(fn, setFrm) {
    HAniS.resetZoom();
    processFOF(fn, setFrm);
  }

  var processFOF = function(fn, setFrm) {
    var i,j,k,m,n;
    var req = new XMLHttpRequest();
    var st, stt, sto, stx, hro, hro1, hro2;
    var fofext = null;
    var gotHotspot = false;
    prevNumHotzones = numHotzones;
    gotImages = false;
    wasLooping = isLooping;
    setIsLooping(false);
    if (setFrm) setCurrentFrame(begFrame);
    numFrames = 0;
    numHotzones = 0;
    backFiles = new Array();
    hoverzones = null;
    sliceCoords = null;
    redirectList = [];
    redirect = null;
    while (divanim.hasChildNodes()) {
      divanim.removeChild(divanim.lastChild);
    }
    numHoverzones = 0;
    gotHoverzones = false;
    doHoverzones = false
    doingHiResZoom = false;
    hiResBase = null;
    hiResBaseName = null;
    hiResOlay = null;
    hiResOlayName = [];
    hiResOlayIndex = -1;
    overlayFiles = new Array();
    var doTimes = false;
    if (extrapTimes == null || extrapTimesTemplate != null) {
      extrapTimes = new Array();
    }
    info("new FOF="+fn);
    if (isExtrap) {
      if (extrap != null) {
        toggleButton(extrap, true, "extrap on");
        divall.style.cursor = "default";
        setIsLooping(wasLooping);
        extrapMode = -1;
      }
      isExtrap = false;
      drawLines();
    }

    if (isMark) {
      if (mark != null) {
        toggleButton(mark, true, "mark on");
        divall.style.cursor = "default";
        setIsLooping(wasLooping);
        markMode = -1;
      }
      isMark = false;
      drawLines();
    }

    req.onload = function() {
      var txt = this.responseText.split("\n");
      for (i=0; i<txt.length; i++) {
        if (txt[i].length < 2) continue;
        st = txt[i].trim();

        info("FOF: "+st);
        if (st.indexOf("#") == 0) continue;

        if (st.indexOf("frame_labels") == 0) {
          hro = st.split("=");
          frameLabels = hro[1].split(",");
          continue;
        }

        if (st.indexOf("redirect") == 0) {
          hro = st.split("=");
          hro2 = hro[1].split(",");
          redirect = hro2[0].trim();
          for (j=1; j<hro2.length; j++) {
            var m = parseInt(hro2[j],10) - 1;
            redirectList[m] = true;
          }
          continue;
        }

        if (st.indexOf("slice") == 0) {
          hro = st.split("=");
          hro2 = hro[1].split(",");
          sliceCoords = [];
          for (j=0; j<hro2.length; j++) {
            sliceCoords[j] = parseInt(hro2[j],10);
          }
          // convert to width and height
          sliceCoords[2] = sliceCoords[2] - sliceCoords[0] + 1;
          sliceCoords[3] = sliceCoords[3] - sliceCoords[1] + 1;
          if (sliceCoords.length > 4) {
            sliceCoords[6] = sliceCoords[6] - sliceCoords[4] + 1;
            sliceCoords[7] = sliceCoords[7] - sliceCoords[5] + 1;
          }
          continue;
        }

        if (st.indexOf("fof_extension") == 0) {
          hro = st.split("=");
          fofext = hro[1].trim();
          continue;
        }

        if (st.indexOf("probe_table") == 0) {
          hro = st.split("=");
          gotTable = false;
          getProbeTable(hro[1].trim());
          continue;
        }

        if (st.indexOf("image_base") == 0) {
          hro = st.split("=");
          imageBase = hro[1].trim();
          continue;
        }

        if (st.indexOf("overlay_base") == 0) {
          hro = st.split("=");
          overlayBase = hro[1].trim();
          continue;
        }

        if (st.indexOf("high_res_basemap") == 0) {
          hro = st.split("=");
          hro2 = hro[1].split(",");
          hiResBaseName = new Array();
          for (j=0; j<hro2.length; j++) {
            hiResBaseName[j] = hro2[j].trim();
            if (imageBase != null) {
              hiResBaseName[j] = imageBase+hiResBaseName[j];
            }
          }
          doingHiResZoom = true;
          continue;
        }


        if (st.indexOf("high_res_overlay") == 0) {
          hro = st.split("=");
          hro1 = hro[1].split("&");
          for (k=0; k<hro1.length; k++) {
            hro2 = hro1[k].split(",");
            // hiResOlayName[olayIndex][hiResIndex]
            hiResOlayIndex = parseInt(hro2[0].trim(),10)-1;
            hiResOlayName[hiResOlayIndex] = new Array();
            for (j=1; j<hro2.length; j++) {
              hiResOlayName[hiResOlayIndex][j-1] = hro2[j].trim();
              if (!endsWith(hiResOlayName[hiResOlayIndex][j-1], "/")) {
                if (overlayBase != null) {
                  hiResOlayName[hiResOlayIndex][j-1] =
                             overlayBase+hiResOlayName[hiResOlayIndex][j-1];
                } else if (imageBase != null) {
                  hiResOlayName[hiResOlayIndex][j-1] =
                             imageBase+hiResOlayName[hiResOlayIndex][j-1];
                }
              }
            }
            doingHiResZoom = true;
          }
          continue;
        }

        if (st.indexOf("high_res_zoom") == 0) {
          hro = st.split("=");
          hro2 = hro[1].split(",");
          zoomFactors = new Array();
          for (j=0; j<hro2.length; j++) {
            zoomFactors[j] = parseFloat(hro2[j].trim());
          }
          setzoomFactorIndex();
          continue;
        }

        if (st.indexOf("hoverzone") == 0) {
          fetchHoverzone(st);
          continue;
        }

        if (st.indexOf("hotzone") == 0) {
          fetchHotzone(st);
          continue;
        }

        if (st.indexOf("hotspot") == 0) {
          if (!gotHotspot) {
            hotspots = null;
            numHotspots = 0;
            gotHotspot = true;
          }
          fetchHotspot(st);
          continue;
        }

        if (st.indexOf("toggle_onoff") == 0) {
          hro = st.split("=");
          toggleDefs = hro[1].split(",");
          continue;
        }

        if (st.indexOf("coordinates") == 0) {
          hro = st.split("=");
          parseCoordinates(hro[1]);
          continue;
        }

        if (st.indexOf("map_scale") == 0) {
          hro = st.split("=");
          hro2 = hro[1].split(",");
          distXScale = parseFloat(hro2[0]);
          if (hro2.length > 1) {
            distYScale = parseFloat(hro2[1]);
          } else {
            distYScale = distXScale;
          }
          continue;
        }

        if (st.indexOf("times") == 0) {
          hro = st.split("=");
          hro2 = hro[1].split(",");
          for (i=0; i<hro2.length; i++) {
            extrapTimes[i] = parseInt(hro2[i].trim(), 10);
          }
          doTimes = true;
          continue;
        }

        if (extrapTimesTemplate != null) {
          var ptimes = st.match(extrapTimesTemplate);
          if (ptimes != null && ptimes.length == 2) {
            extrapTimes.push(parseInt(ptimes[1],10));
            info("+ + + Extrap time = "+ptimes[1]);
            doTimes = true;
          } else {
            info("+ + + Cannot parse times using "+extrapTimesTemplate+" from line: "+st);
          }

        }

        k = st.indexOf(" ");
        if (k < 0) k = st.length;
        backFiles[numFrames] = st.substring(0,k).trim();
        if (imageBase != null) {
          backFiles[numFrames] = imageBase+backFiles[numFrames];
        }

        st = st.substring(k+1);
        m = st.indexOf('"');
        if (m >= 0) {
          if (frameLabels == null) {
            frameLabels = new Array();
          }
          k = st.substring(m+1);
          m = k.indexOf('"');
          if (m >= 0) {
            frameLabels[numFrames] = k.substring(0,m);
            if (numFrames == 0) {
              frameLabelField.innerHTML = frameLabels[0];
            }
          }
        }

        overlayFiles[numFrames] = new Array();

        m = st.indexOf("overlay");
        if (m >= 0) {
          stt = st.substring(m);
          m = stt.indexOf("=");
          if (m < 0) {
            info("Cannot find a = sign in "+stt);
          }
          sto = stt.substring(m+1);
          stt = sto.trim();
          sto = stt.split(",");
          numOverlays = sto.length;
          info("numOverlays set to "+numOverlays);

          for (n=0; n<sto.length; n++) {
            overlayFiles[numFrames][n] = sto[n].trim();
            if ((redirect != null) && redirectList[n]) {
              overlayFiles[numFrames][n] =
                   (redirect+overlayFiles[numFrames][n]).trim();
            } else if (overlayBase != null) {
              overlayFiles[numFrames][n] =
                   (overlayBase+overlayFiles[numFrames][n]).trim();
            } else if (imageBase != null) {
              overlayFiles[numFrames][n] =
                   (imageBase+overlayFiles[numFrames][n]).trim();
            }
          }
        }

        numFrames = numFrames + 1;
      }

      if (doTimes) makeTimes(extrapTimes);

      if (fofext != null) {
        var reqx = new XMLHttpRequest();
        reqx.onload = function() {
          var txtx = this.responseText.split("\n");
          for (i=0; i<txtx.length; i++) {
            if (txtx[i].length < 2) continue;
            stx = txtx[i].trim();

            info("ext_FOF: "+stx);
            if (stx.indexOf("#") == 0) continue;

            if (stx.indexOf("hotzone") == 0) {
              fetchHotzone(stx);
              continue;
            }
            if (stx.indexOf("hotspot") == 0) {
              fetchHotspot(stx);
              continue;
            }
          }

          readyToLoad();
        }

        reqx.onerror = function() {
          readyToLoad();
        }

        reqx.open("get", fofext+"?"+Math.round(Math.random()*100000), true);
        reqx.send();

      } else {
        readyToLoad();
      }
    }


    lastFOF = fn;
    req.open("get", fn+"?"+Math.round(Math.random()*100000), true);
    req.send();

  }

  function readyToLoad() {
    if (numHotzones != 0) {
      setHotzoneVis(numHotzones, "visible");
    } else if (prevNumHotzones != 0) {
      setHotzoneVis(prevNumHotzones, "hidden");
    }

    fetchImages = true;
    loadImages();
  }

  this.setWindowSize = function(w, h) {
    if (divall == null) return;
    divall.style.height = h+"px";
    divall.style.width = w+"px";
    divall.style.padding = "0px";
    var dcn = (divcont != null) ? divcont.offsetHeight : 0;
    var dcnb = (divconb != null) ? divconb.offsetHeight : 0;
    var dtog = (divtog != null) ? divtog.offsetHeight : 0;
    var doly = (divolay != null) ? divolay.offsetHeight : 0;

    var hgt = h - dcn - dcnb - dtog - numdivolay*doly;
    prefHgt = hgt;
    prefWid = w;
    if (hgt > imgHeight) hgt = imgHeight;
    var wid = hgt * imgWidth / imgHeight;
    if (wid > w) {
      wid = w;
      hgt = w * imgHeight / imgWidth;
    }
    hgt = Math.floor(hgt);
    wid = Math.floor(wid);

    if (needSizes) return;
    resizeWindow(wid, hgt);
    drawLines();
  }

  function resizeWindow(w,h) {
    if (needSizes) return;
    imgCan.height = h;
    imgCan.width = w;
    drwCan.height = h;
    drwCan.width = w;
    topCan.height = h;
    topCan.width = w;
    aeCan.height =h;
    aeCan.width =w;

    ctxae = aeCan.getContext("2d");

    canW = w;
    canH = h;
    progX = canW/2 - 100;
    progY = canH/2 - 10;
    canXScale = 1.0 * imgWidth / canW;
    canYScale = 1.0 * imgHeight / canH;

    if (winFirst) {
      winFirst = false;
      if (initCall != null) initCall();
    }
    divcan.style.height = canH+"px";
    divcan.style.width = canW+"px";
    divanim.style.height = canH+"px";
    divanim.style.width = canW+"px";
    divtop.style.height = canH+"px";
    divtop.style.width = canW+"px";
    makeToggles(numFrames, false);
  }

  function getzipback() {
    var cnt = 0;
    for (var i=0; i<numFrames; i++) {
      let ii = i;
      var zbfn = (imageBase == null) ? backFiles[i] : backFiles[i].substring(backFiles[i].lastIndexOf("/")+1);
      var fzip = zipFile.file(zbfn);
      if (fzip) {
        fzip.async("arraybuffer").then(function(data){
          var imgBlob = new Blob([data]);
          backFilesUrl[ii] = URL.createObjectURL(imgBlob);
          cnt++;
          if (cnt >= numFrames) getzipoverlay();
        }, function() {
          backFilesUrl[ii] = null;
          cnt++;
          if (cnt >= numFrames) getzipoverlay();
        });
      } else {
        backFilesUrl[ii] = null;
        cnt++;
        if (cnt >= numFrames) getzipoverlay();
      }
    }
  }

  function getzipoverlay() {
    var cnt = 0;
    if (numOverlays == 0) {
      getImages();
      return;
    }
    for (var i=0; i<numFrames; i++) {
      overlayFilesUrl[i] = [];
      for (var j=0; j<numOverlays; j++) {
        let ii = i;
        let jj = j;
        var zofn = (overlayBase == null && imageBase == null) ? overlayFiles[i][j] : overlayFiles[i][j].substring(overlayFiles[i][j].lastIndexOf("/")+1);
        var fzip = zipFile.file(zofn);
        if (fzip) {
          fzip.async("arraybuffer").then(function(data){
            var imgBlob = new Blob([data]);
            overlayFilesUrl[ii][jj] = URL.createObjectURL(imgBlob);
            cnt++;
            if (cnt >= numFrames*numOverlays) getImages();
          }, function() {
            overlayFilesUrl[ii][jj] = null;
            cnt++;
            if (cnt >= numFrames*numOverlays) getImages();
          });
        } else {
          overlayFilesUrl[i][j] = null;
          cnt++;
          if (cnt >= numFrames*numOverlays) getImages();
        }
      }
    }
  }

  function loadImages() {
    if (usingZip) {
      // read the zip file, get the blobs for each image as a URL
      var xhrzip = new XMLHttpRequest();
      var zfn = fofBase == null ? zipFilename : fofBase+zipFilename;
      if (!zipStatic) zfn = zfn + "?"+Math.round(Math.random()*100000)
      xhrzip.open("GET", zfn , true);
      xhrzip.responseType = "arraybuffer";
      backFilesUrl = [];
      overlayFilesUrl = [];
      xhrzip.addEventListener("progress", function(evt) {
        if (evt.lengthComputable) {
          drawProgress("Loading ZIP file", evt.total, evt.loaded);
        }
      } );
      xhrzip.onload = function(ez) {
        var zippy = new JSZip();
        /**@nocollapse*/
        zippy.loadAsync(this.response).then( function(zip){
          zipFile = zip;
          getzipback();
        }, function() {
         info("Unable to read ZIP file: "+zipFilename);
        });
      }
      xhrzip.send();
    } else {
      getImages();
    }
  }

  function getImages() {
    var i,j,k;
    if (cdLoading != undefined) cdLoading(false);
    var olayNotStatic = "?"+Math.round(Math.random()*10000);
    var loadOrder = [];
    loadOrder[0] = begFrame;
    for (i=1; i<numFrames; i++) {
      loadOrder[i] = (i <= begFrame) ? i-1 : i;
    }
    backImages = new Array(numFrames);
    imgCount = imgGotCount = 0;
    useProgress = showProgress;
    if (isSetframe) setframe.max = numFrames;
    if (needSizes && (isBaseEnh || isOverlayEnh)) {
      origCan = new Array();
      enhCan = new Array();
      ctxe = new Array();
      origIDd = new Array();
      enhID = new Array();
    }
    if (enhance != null) enhance.selectedIndex = 0;

    for (var ilf=0; ilf<numFrames; ilf++){
      i = loadOrder[ilf];
      if (needSizes && (isBaseEnh || isOverlayEnh)) {
        origCan[i] = make("canvas");
        enhCan[i] = make("canvas");
      }
      backImages[i] = new Image();
      backImages[i].gotit = false;
      backImages[i].frameNum = i;
      backImages[i].onerror = function() {
        imgGotCount++;
      }

      backImages[i].onload = function() {

        if (needSizes) {
          imgHeight = this.height;
          imgWidth = this.width;
          hImage = imgHeight;
          wImage = imgWidth;
          needSizes = false;
          if (userWindow) {
            if (useDiv) {
              HAniS.setWindowSize(divall.offsetWidth, divall.offsetHeight);
            } else {
              resizeWindow(canW, canH);
            }
          } else {
            resizeWindow(imgWidth, imgHeight);
          }
          if (isInitialZoom) {
            if (initialZoom == "auto") {
              initialZoom = (canXScale + canYScale) / 2.0;
              initialX = -1
            }
            hImage = hImage / initialZoom;
            wImage = wImage / initialZoom;
            if (initialX < 0) {
              initialX = imgWidth / 2.0;
              initialY = imgHeight / 2.0;
            }
            xMove = Math.round(initialX - imgWidth/2/initialZoom);
            yMove = Math.round(initialY - imgHeight/2/initialZoom);
            zoomXFactor = zoomYFactor = initialZoom;
            if (initialZoom != 1.0) {
              enableZooming = false;
              HAniS.toggleZooming();
            }
          }
        }

        if (usingZip && backFilesUrl[this.frameNum] != null) {
          URL.revokeObjectURL(backFilesUrl[this.frameNum]);  // get rid of it...
        }

        if (numHoverzones > 0 && !gotHoverzones) {
          hoverCan.height = imgHeight;
          hoverCan.width = imgWidth;
          ctxh = hoverCan.getContext("2d");
          ctxh.clearRect(0,0, canW, canH);
          for (var l=0; l<hoverzones.length; l++) {
            var hz = hoverzones[l].ctx.getImageData(0,0,
               hoverzones[l].width, hoverzones[l].height).data;
            var tar = ctxh.getImageData(
               hoverzones[l].xmin, hoverzones[l].ymin,
               hoverzones[l].width, hoverzones[l].height);
            for (var m=3; m<hz.length; m+=4) {
              if (hz[m] != 0) {
                tar.data[m] = (l+1);
              }
            }
            ctxh.putImageData(tar, hoverzones[l].xmin, hoverzones[l].ymin);
          }
          if (!enableZooming) doHoverzones = true;
          if (zoomXFactor == zoomXBase) doHoverzones = true;
          gotHoverzones = true;
        }

        if (fetchImages) {
          fetchImages = false;
          ctx = imgCan.getContext("2d");
          ctx.imageSmoothingEnabled = enableSmoothing;
          ctx.fillStyle="blue";
          ctx.font="bold 20px Arial";
          ctxd = drwCan.getContext("2d");
          ctxd.imageSmoothingEnabled = enableSmoothing;
          toggleFrames = new Array(numFrames);
          makeToggles(numFrames, true);
          if (refLooping && !wasLooping) {
            setCurrentFrame(findFrame(begFrame));
          } else {
            setCurrentFrame(findFrame(curFrame));
          }

          if (numOverlays > 0) {
            overlayImages = new Array(numFrames);
            for (var klf=0; klf<numFrames; klf++) {
              k = loadOrder[klf];
              overlayImages[k] = new Array(numOverlays)
              for (j=0; j<numOverlays; j++) {
                overlayImages[k][j] = new Image();
                overlayImages[k][j].gotit = false;
                overlayImages[k][j].frameNum = k;
                overlayImages[k][j].overlayNum = j;
                overlayImages[k][j].onerror = function() {
                  imgGotCount++;
                }
                overlayImages[k][j].onload = function() {
                  this.gotit = true;
                  var f = this.frameNum;
                  imgGotCount++;
                  drawImageProgress();
                  if (chkImageSize && (this.height != imgHeight || this.width != imgWidth)) {
                    this.gotit = false;
                    info("Bad image size:"+this.src);
                  } else {
                    if (this.overlayNum == missTog-1) toggleFrames[this.frameNum] = 0;
                  }

                  if (isOverlayEnh && this.gotit && this.overlayNum == overlayEnhNum) {
                    origCan[f].height = this.height;
                    origCan[f].width = this.width;
                    ctxo = origCan[f].getContext("2d");
                    ctxo.drawImage(this, 0, 0);
                    origIDd[f] = ctxo.getImageData(0,0,this.width,this.height).data;

                    enhCan[f].height = this.height;
                    enhCan[f].width = this.width;
                    ctxe[f] = enhCan[f].getContext("2d");
                    ctxe[f].drawImage(this, 0, 0)
                    enhID[f] = ctxe[f].getImageData(0,0,this.width,this.height);

                    overlayImages[f][this.overlayNum] = null;
                    overlayImages[f][this.overlayNum] = enhCan[f];
                    overlayImages[f][this.overlayNum].gotit = true;

                  } else if (this.gotit && overlayCheck[this.overlayNum].hotspotIndex) {

                      var cnt = 0;
                      var tpx, tpy, tph, tpon, tpv;
                      var tpcan = make("canvas");
                      tpcan.height = this.height;
                      tpcan.width = this.width;
                      var ctxtp = tpcan.getContext("2d");
                      ctxtp.drawImage(this,0,0);
                      var tpdd = ctxtp.getImageData(0,0,this.width, this.height).data;
                      var npt = tpdd.length;
                      tpon = this.overlayNum;
                      var k;
                      for (k=0; k<npt; k=k+4) {
                        if (tpdd[k] != 0) {
                          if (numHotspots == 0) hotspots = new Array();
                          cnt = cnt + 1;
                          tpv = tpdd[k];
                          tpy = Math.floor((k/4)/tpcan.width);
                          tpx = Math.floor((k/4) % tpcan.width);
                          hotspots[numHotspots] = new Hotspot(tpx, tpy, "spritefn",tpv, "pan", null, null, tpon, f, null);
                         numHotspots++;
                        }
                      }

                  } else if (this.gotit && doTransparency) {

                    if (!useTransparencyList || (useTransparencyList &&
                         transparencyList[this.overlayNum])) {
                      var tpcan = make("canvas");
                      tpcan.height = this.height;
                      tpcan.width = this.width;
                      var ctxtp = tpcan.getContext("2d");
                      ctxtp.drawImage(this,0,0);
                      var tpdata = ctxtp.getImageData(0,0,this.width, this.height);
                      var tpdd = tpdata.data;
                      var npt = tpdd.length;
                      var k;
                      for (k=0; k<npt; k=k+4) {
                        if (tpdd[k] === transRed && tpdd[k+1] === transGreen &&
                               tpdd[k+2] === transBlue) {

                          tpdd[k+3] = 0;
                        }
                      }
                      ctxtp.putImageData(tpdata,0,0);
                      overlayImages[f][this.overlayNum] = null;
                      overlayImages[f][this.overlayNum] = tpcan;
                      overlayImages[f][this.overlayNum].gotit = true;
                    }
                  }

                  if (this.gotit && overlayTop[this.overlayNum]) {
                    ctxtop.clearRect(0,0,canW, canH);
                    ctxtop.drawImage(overlayImages[f][this.overlayNum],0,0,canW,canH);

                  } else if (this.gotit && overlaySlice[this.overlayNum] && (sliceCoords != null)) {
                    var tpcan = make("canvas");
                    tpcan.height = imgHeight;
                    tpcan.width = imgWidth;
                    if (sliceCoords.length < 8) {
                      sliceCoords[4] = 0;
                      sliceCoords[5] = 0;
                      sliceCoords[6] = imgWidth;
                      sliceCoords[7] = imgHeight;
                    }
                    var ctxtp = tpcan.getContext("2d");
                    ctxtp.imageSmoothingEnabled=sliceSmoothing;
                    ctxtp.drawImage(this,sliceCoords[0], sliceCoords[1], sliceCoords[2],sliceCoords[3],sliceCoords[4],sliceCoords[5],sliceCoords[6], sliceCoords[7]);
                    overlayImages[f][this.overlayNum] = null;
                    overlayImages[f][this.overlayNum] = tpcan;
                    overlayImages[f][this.overlayNum].gotit = true;
                    if (preserveIndex !=null && preserveIndex[this.overlayNum]) {
                      overlayImages[f][this.overlayNum].origImage = this;
                    }
                  }
                  if (usingZip && overlayFilesUrl[f][this.overlayNum] != null) {
                    URL.revokeObjectURL(overlayFilesUrl[f][this.overlayNum]);
                  }
                }

                var ofn = overlayFiles[k][j];
                if (useForAll[j+1] != undefined) {
                  var ufai = useForAll[j+1];
                  if (ufai >= numFrames) ufai = numFrames-1;
                  ofn = overlayFiles[ufai][j];
                }

                if (overlayStatic != undefined && !overlayStatic[j]) {
                  ofn = ofn+olayNotStatic;
                }
                overlayImages[k][j].src = ofn;
                if (usingZip && (zipOnly || overlayFilesUrl[k][j]) ) {
                  overlayImages[k][j].src = overlayFilesUrl[k][j];
                } else {
                  overlayImages[k][j].src = ofn;
                }
                imgCount++;
              }
            }
          }

          if (hiResBaseName != null) {
            hiResBase = new Array();
            for (j=0; j<hiResBaseName.length; j++) {
              hiResBase[j] = new Image();
              hiResBase[j].gotit = false;
              hiResBase[j].onerror = function(e) {
                imgGotCount++;
              }
              hiResBase[j].onload = function(e) {
                e.currentTarget.gotit = true;
                imgGotCount++;
                drawImageProgress();
              }

              hiResBase[j].src = hiResBaseName[j];
              imgCount++;
            }
          }

          if (hiResOlayName != null && hiResOlayName.length > 0) {
            hiResOlay = new Array();
            // hiResOlay[olayIndex][hiResIndex][frame]
            for (j=0; j<hiResOlayName.length; j++) {
              if (hiResOlayName[j] == undefined) continue;
              hiResOlay[j] = []

              for (k=0; k<hiResOlayName[j].length; k++) {
                hiResOlay[j][k] = [];

                for (m=0; m<numFrames; m++) {
                  if (endsWith(hiResOlayName[j][k], "/")) {
                    var u = overlayFiles[m][j].lastIndexOf("/");
                    var hron = overlayFiles[m][j].substring(0,u+1)+hiResOlayName[j][k]+overlayFiles[m][j].substring(u+1);
                    hiResOlay[j][k][m] = new Image();
                    hiResOlay[j][k][m].gotit = false;
                    hiResOlay[j][k][m].onerror = function(e) {
                      imgGotCount++;
                      drawImageProgress();
                    }
                    hiResOlay[j][k][m].onload = function(e) {
                      e.currentTarget.gotit = true;
                      imgGotCount++;
                      drawImageProgress();
                    }
                    hiResOlay[j][k][m].src = hron;
                    imgCount++;

                  } else {
                    if (m === 0) {
                      hiResOlay[j][k][0] = new Image();
                      hiResOlay[j][k][0].gotit = false;
                      hiResOlay[j][k][0].onerror = function(e) {
                        imgGotCount++;
                        drawImageProgress();
                      }
                      hiResOlay[j][k][0].onload = function(e) {
                        e.currentTarget.gotit = true;
                        imgGotCount++;
                        drawImageProgress();
                      }

                      hiResOlay[j][k][0].src = hiResOlayName[j][k];
                      imgCount++;

                    } else {
                      hiResOlay[j][k][m] = hiResOlay[j][k][0];
                    }
                  }
                }
              }
            }
          }

        }

        if ((!chkImageSize && this.height != 0 && this.width != 0) || this.height == imgHeight && this.width == imgWidth) {
          this.gotit = true;
          if (missTog == 0) toggleFrames[this.frameNum] = 0;

        } else {
          this.gotit = false;
          info("Image has invalid size:"+this.src);
        }

        if (isBaseEnh && this.gotit) {
          var f = this.frameNum;
          origCan[f].height = this.height;
          origCan[f].width = this.width;
          ctxo = origCan[f].getContext("2d");
          ctxo.drawImage(this, 0, 0);
          origIDd[f] = ctxo.getImageData(0,0,this.width, this.height).data;

          enhCan[f].height = this.height;
          enhCan[f].width = this.width;
          ctxe[f] = enhCan[f].getContext("2d");
          ctxe[f].drawImage(this, 0, 0)
          enhID[f] = ctxe[f].getImageData(0,0,this.width, this.height);

          backImages[f] = null;
          backImages[f] = enhCan[f];
          backImages[f].gotit = true;
        }

        imgGotCount++;
        drawImageProgress();

        if (!keepZoom) HAniS.resetZoom();
        gotImages = true;

        if (!isRunning) {
          isRunning = true;
          isLooping = false;
          run();
        }

        setIsLooping(wasLooping);
      }

      var bfn = backFiles[i];
      if (useForAll[0] != undefined) {
        var ufai = useForAll[0];
        if (ufai >= numFrames) ufai = numFrames-1;
        bfn = backFiles[ufai];
      }
      if (!backStatic) bfn = bfn+olayNotStatic;
      if (usingZip && (zipOnly || backFilesUrl[i]) ) {
        backImages[i].src = backFilesUrl[i];
      } else {
        backImages[i].src = bfn;
      }
      imgCount++;
    }


    showTip = false;
    showDistance = false;
    drawLines();
  }

  function configButton(button, name, defOn, defOff, tip, clsnam) {
    button.label_on = defOn;
    button.label_off = defOff;
    if (useCN) button.className = clsnam;
    var wid = null;
    var lab = configValues[name+"_labels"];
    if (lab == null) lab = configValues[name+"_label"];
    if (lab != null) {
      var ssl = lab.split(",");
      button.label_on = ssl[0].trim();
      if (ssl.length > 1) button.label_off = ssl[1].trim();
      if (ssl.length > 2) wid = "width:"+ssl[2].trim()+"px;"
    }

    if (tip != null) button.title = tip;

    button.innerHTML = button.label_on;

    var stycv = configValues[name+"_style"];
    var styOn = stycv;
    var styOff = stycv
    if (stycv != null) {
      var stysp = stycv.split("&");
      if (stysp.length > 1) {
        styOn = stysp[0];
        styOff = stysp[1];
      }
    }

    button.style_on = (wid!=null?wid:"")+buttcss+styOn;
    button.style_off = (wid!=null?wid:"")+buttcss+styOff;
    if (button.style_on != null && button.style_on.length > 1) {
      button.setAttribute("style", button.style_on);
    } else {
      button.style_on = null;
      button.style_off = null;
    }
  }

  function doMakeControls(control, isTop) {

    var i,k, tips, cv;
    buttcss = configValues["buttons_style"];
    if (buttcss == null) buttcss = "";
    var tooltips = null, wid = null, sty = null;

    if (isTop) {
      tooltips = configValues["controls_tooltip"];
    } else {
      tooltips = configValues["bottom_controls_tooltip"];
    }

    var c = control.split(",");

    if (tooltips != null) {
      tips = tooltips.split(",");
      if (tips.length != c.length) {
        info("!!! Number of tooltips must equal number of controls!");
        tooltips = null;
        tips = null;
      }
    }

    var ssl, mytip;
    for (i=0; i<c.length; i++) {
      if (tooltips != null) {
        mytip = tips[i];
      } else {
        mytip = null;
      }

      var cstr = c[i].trim();
      var cslash = cstr.indexOf("/");

      if (cslash === 0 || i === 0) {
        if (cslash === 0) cstr = cstr.substring(1);
        divcon = make("div");
        divcon.align=divalign;
        cv = configValues["controls_style"];
        if (cv != null) {
          divcon.setAttribute("style",cv);
        } else {
          divcon.setAttribute("style","position:relative;padding:2px;width:100%;");
        }

        if (isTop) {
          divall.insertBefore(divcon, divcan);
          divcont = divcon;

        } else {
          cv = configValues["bottom_controls_style"];
          if (cv != null) {
            divcon.setAttribute("style",cv);
          }
          divall.appendChild(divcon);
          divconb = divcon;
        }
      }

      if (cstr.trim() == "enhance") {
        enhance = make("select");
        sty = configValues["enhance_style"];
        if (sty != null) enhance.setAttribute("style", sty);
        if (mytip != null) enhance.title = mytip;
        var opt = make("option");
        opt.innerHTML = "Pick Enhancement";
        opt.value = -1;
        enhance.add(opt);
        enhance.addEventListener("change",HAniS.doEnhance,false);
        isBaseEnh = !isOverlayEnh;
        if (isBaseEnh) {
          enhance.disabled = false;
        } else {
          enhance.disabled = true;
        }
        divcon.appendChild(enhance);
      }

      if (cstr == "startstop") {
        startstop = make("button");
        configButton(startstop, "startstop", "Start", "Stop", mytip, "startstop start");
        startstop.addEventListener("click",HAniS.toggleIsLooping,false);
        divcon.appendChild(startstop);
      }

      if (cstr == "looprock") {
        looprock = make("button");
        configButton(looprock, "looprock", "Rock", "Loop", mytip, "looprock rock");
        direction = 1;
        looprock.addEventListener("click",HAniS.toggleLoopRock,false);
        divcon.appendChild(looprock);
      }

      if (cstr == "loophalt") {
        loophalt = make("button");
        configButton(loophalt, "loophalt", "Halt", "Loop", mytip, "loophalt halt");
        haltMe = false;
        loophalt.addEventListener("click",HAniS.toggleLoopHalt,false);
        divcon.appendChild(loophalt);
      }

      if (cstr == "extrap") {
        extrap = make("button");
        configButton(extrap, "extrap", "Enable Extrap", "Disable Extrap", mytip, "extrap");
        extrap.addEventListener("click",HAniS.toggleExtrap,false);
        divcon.appendChild(extrap);
      }

      if (beginsWith(cstr,"location")) {
        location = make("button");
        configButton(location, "location", "Show Loc", "Hide Loc", mytip, "location");
        doLocation = true;

        if (cstr.indexOf("/on") > 0) {
          toggleButton(location, false, "location off");
          showLocation = true;
          prevCursor.push(divall.style.cursor);
          divall.style.cursor = locCursor;
        } else {
          showLocation = false;
          divall.style.cursor = prevCursor.pop();;
        }

        location.addEventListener("click", function() {
          showLocation = !showLocation;
          if (showLocation) {
            toggleButton(location, false, "location off");
            prevCursor.push(divall.style.cursor);
            divall.style.cursor = locCursor;
          } else {
            toggleButton(location, true, "location on");
            divall.style.cursor = prevCursor.pop();;
          }
          drawLines();

        }, false);
        divcon.appendChild(location);
      }

      if (cstr == "distance") {
        distance = make("button");
        configButton(distance, "distance", "Show dist", "Hide dist", mytip,"distance");

        distShift = false;
        doDistance = false;

        distance.addEventListener("click", function() {
          doDistance = !doDistance;
          if (doDistance) {
            toggleButton(distance, false, "distance off");
            prevCursor.push(divall.style.cursor);
            divall.style.cursor = distCursor;
          } else {
            toggleButton(distance, true, "distance on");
            divall.style.cursor = prevCursor.pop();
          }
        }, false);
        divcon.appendChild(distance);
      }

      if (beginsWith(cstr, "probe")) {
        probe = make("button");
        configButton(probe, "probe", "Show probe", "Hide probe", mytip, "probe");

        if (cstr.indexOf("/on") > 0) {
          toggleButton(probe, false, "probe off");
          showProbe = true;
          prevCursor.push(divall.style.cursor);
          divall.style.cursor = probeCursor;
        } else {
          showProbe = false;
          divall.style.cursor = prevCursor.pop();
        }

        probe.addEventListener("click", function() {
          showProbe = !showProbe;
          if (showProbe) {
            toggleButton(probe, false, "probe off");
            prevCursor.push(divall.style.cursor);
            divall.style.cursor = probeCursor;
          } else {
            toggleButton(probe, true, "probe on");
            divall.style.cursor = prevCursor.pop();
          }
        }, false);

        divcon.appendChild(probe);
      }

      if (cstr == "mark") {
        mark = make("button");
        configButton(mark, "mark", "Mark", "Stop Mark", mytip, "mark");
        drawPrompts = true;
        mark.addEventListener("click", function() {
          isMark = !isMark;
          markPoints = [];
          if (isMark) {
            markMode = 0;
            toggleButton(mark, false, "mark off");
            if (zoom != null) zoom.disabled = true;
            prevCursor.push(divall.style.cursor);
            divall.style.cursor = markCursor;
          } else {
            toggleButton(mark, true, "mark on");
            if (zoom != null) zoom.disabled = false;
            markMode = -1;
            divall.style.cursor = prevCursor.pop();
          }
          drawLines();
        }, false);
        markSaveTB = new TextBox("black","white",markFont, "black",20,5,5);
        markTB = new TextBox("black","white",markFont, "black",20,5,5);
        divcon.appendChild(mark);
      }

      if (cstr == "refresh") {
        refresh = make("button");
        configButton(refresh, "refresh", "Refresh", null, mytip, "refresh");

        refresh.addEventListener("click",HAniS.reloadFOF,false);
        divcon.appendChild(refresh);
      }

      if (cstr == "restore") {
        restore = make("button");
        configButton(restore, "restore", "Restore", null, mytip, "restore");
        restore.addEventListener("click",function(e) {
          for (var i=0; i<overlayCheck.length; i++) {
            overlayCheck[i].checked = overlayCheck[i].restorableState;
          }
          for (var i=0; i<overlayCheck.length; i++) {
            resetLinks(i);
          }
        },false);
        divcon.appendChild(restore);
      }

      if (cstr == "setframe" ) {
        setframe= make("input");
        if (useCN) setframe.className = "setframe";
        setframe.type = "range";
        setframe.min = 1;
        setframe.max = 100;
        setframe.value=1;
        setframeLabel = "Frame #*";
        if (mytip != null) setframe.title = mytip;
        cv = configValues["setframe_label"];
        if (cv != null) {
          setframeLabel = cv;
        }

        setframeLabelSpan = make("span");
        setframeLabelSpan.innerHTML = setframeLabel.replace("*",1);

        sty ="display:block;vertical-align:middle;"+configValues["setframe_style"];
        setframe.setAttribute("style", sty);

        var dsf = make("span");
        dsf.setAttribute("style","display:inline-block;vertical-align:top;");
        setframeLabelSpan.setAttribute("style","display:block;vertical-align:middle;"+configValues["setframe_label_style"]);
        dsf.appendChild(setframeLabelSpan);
        dsf.appendChild(setframe);

        divcon.appendChild(dsf);
        setframe.addEventListener("input", function(e) {
          setIsLooping(false);
          var sf = parseInt(setframe.value,10)-1;
          if (toggleFrames[sf] >= 0) {
            setCurrentFrame(sf);
            setframeLabelSpan.innerHTML = setframeLabel.replace("*",(curFrame+1));
          }
          drawIt();
        }, false);
        isSetframe = true;
      }

      if (cstr == "framelabel" ) {
        frameLabelField = make("button");
        if (useCN) frameLabelField.className = "framelabel";
        frameLabelField.innerHTML = "";
        if (mytip != null) frameLabelField.title = mytip;
        frameLabels = null;
        cv = configValues["frame_labels"];
        if (cv != null) {
          frameLabels = cv.split(",");
          frameLabelField.innerHTML = frameLabels[0];
        }
        sty = buttcss+configValues["framelabel_style"];
        if (sty != null) frameLabelField.setAttribute("style", sty);
        var flp = configValues["framelabel_position"];
        if (flp != null) {
          var flpp = flp.split(",");
          frameLabelField.style.position = "absolute";
          frameLabelField.style.left = (divall.offsetLeft + parseInt(flpp[0],10))+"px";
          frameLabelField.style.top = (divall.offsetTop + parseInt(flpp[1],10))+"px";
        }
        divcon.appendChild(frameLabelField);
      }

      if (cstr == "autorefresh" || cstr == 'autorefresh/off') {
        if (refreshTimer != null) clearInterval(refreshTimer);
        autotoggle = make("button");
        configButton(autotoggle, "autorefresh","Disable Auto Refresh","Enable Auto Refresh", mytip, "autorefresh");
        autotoggle.addEventListener("click",HAniS.toggleAutoRefresh,false);
        isAutoRefresh = false;
        divcon.appendChild(autotoggle);
        if (cstr == "autorefresh") {
          HAniS.toggleAutoRefresh();
        } else {
          toggleButton(autotoggle, false, "autotoggle off");
        }
      }

      if (cstr == "backward") {
        backward = make("button");
        configButton(backward, "backward", "<", "<", mytip, "step back");
        backward.addEventListener("click",function() {
          setIsLooping(false);
          incCurrentFrame(-1);
          drawIt();
        }, false);
        divcon.appendChild(backward);
      }

      if (cstr == "forward") {
        forward = make("button");
        configButton(forward, "forward", ">", ">", mytip, "step forward");
        forward.addEventListener("click",function() {
          setIsLooping(false);
          incCurrentFrame(+1);
          drawIt();
        }, false);
        divcon.appendChild(forward);
      }

      if (cstr == "step") {
        var sl = configValues["step_labels"];
        var sl2 = (sl != null) ? sl.split(",") : ["<", ">"];
        backward = make("button");
        if (useCN) backward.className = "step back";
        backward.innerHTML = sl2[0];
        backward.addEventListener("click",function() {
          setIsLooping(false);
          incCurrentFrame(-1);
          drawIt();
        }, false);

        sty = buttcss+configValues["step_style"];
        if (sty != null) backward.setAttribute("style", sty);
        if (mytip != null) backward.title = mytip;
        divcon.appendChild(backward);

        forward = make("button");
        if (useCN) forward.className = "step forward";
        forward.innerHTML = sl2[1];
        forward.addEventListener("click",function() {
          setIsLooping(false);
          incCurrentFrame(+1);
          drawIt();
        }, false);

        if (sty != null) forward.setAttribute("style", sty);
        if (mytip != null) forward.title = mytip;
        divcon.appendChild(forward);
      }

      if (cstr == "first") {
        first = make("button");
        configButton(first, "first", "|<", "|<", mytip, "firstlast first");
        first.addEventListener("click",function() {
          setIsLooping(false);
          setCurrentFrame(findFrame(0));
          drawIt();
        }, false);
        divcon.appendChild(first);
      }

      if (cstr == "last") {
        last = make("button");
        configButton(last, "last", ">|", ">|", mytip, "firstlast last");
        last.addEventListener("click",function() {
          setIsLooping(false);
          setCurrentFrame(findFrame(numFrames - 1));
          drawIt();
        }, false);
        divcon.appendChild(last);
      }

      if (cstr == "firstlast") {
        var sl = configValues["firstlast_labels"];
        var sl2 = (sl != null) ? sl.split(",") : ["<<", ">>"];
        first = make("button");
        if (useCN) first.className = "firstlast";
        first.innerHTML = sl2[0];
        first.addEventListener("click",function() {
          setIsLooping(false);
          setCurrentFrame(findFrame(0));
          drawIt();
        }, false);

        sty = buttcss+configValues["firstlast_style"];
        if (sty != null) first.setAttribute("style", sty);
        if (mytip != null) first.title = mytip;
        divcon.appendChild(first);

        last = make("button");
        if (useCN) last.className = "firstlast";
        last.innerHTML = sl2[1];
        last.addEventListener("click",function() {
          setIsLooping(false);
          setCurrentFrame(findFrame(numFrames - 1));
          drawIt();
        }, false);

        if (sty != null) last.setAttribute("style", sty);
        if (mytip != null) last.title = mytip;
        divcon.appendChild(last);
      }

      if (cstr == "setspeed") {
        setSpeed = make("input");
        if (useCN) setSpeed.className = "setspeed";
        setSpeed.type = "range";
        setSpeed.min = minDwell;
        setSpeed.max = maxDwell;
        setSpeed.value = maxDwell - dwell;
        if (mytip != null) setSpeed.title = mytip;
        sty ="vertical-align:middle;"+configValues["setspeed_style"];
        setSpeed.setAttribute("style", sty);
        setSpeed.addEventListener("input", function(e) {
          dwell = maxDwell - parseInt(setSpeed.value,10) + 30;
          if (dwChanged != undefined) dwChanged(dwell);
          setIsLooping(true);
          drawIt();
        }, false);
        divcon.appendChild(setSpeed);


      } else if (cstr == "speed") {
        var sl = configValues["speed_labels"];
        var sl2 = (sl != null) ? sl.split(",") : ["-","+"];
        slower = make("button");
        if (useCN) slower.className = "speed slower";
        slower.innerHTML = sl2[0];
        slower.addEventListener("click",function() {
          dwell = dwell + stepDwell;
          faster.disabled = false;
          if (dwell > maxDwell) {
            dwell = maxDwell;
            slower.disabled = true;
          }
          if (dwChanged != undefined) dwChanged(dwell);
        }, false);

        sty = buttcss+configValues["speed_style"];
        if (sty != null) slower.setAttribute("style", sty);
        if (mytip != null) slower.title = mytip;
        divcon.appendChild(slower);

        faster = make("button");
        if (useCN) faster.className = "speed faster";
        faster.innerHTML = sl2[1];
        faster.addEventListener("click",function() {
          dwell = dwell - stepDwell;
          slower.disabled = false;
          if (dwell < minDwell) {
            dwell = minDwell;
            faster.disabled = true;
          }
          if (dwChanged != undefined) dwChanged(dwell);
        }, false);
        if (sty != null) faster.setAttribute("style", sty);
        if (mytip != null) faster.title = mytip;
        divcon.appendChild(faster);
      }

      if (cstr == "zoom") {
        zoom = make("button");
        configButton(zoom, "zoom", "Zoom", "Un-zoom", mytip, "zoom off");
        zoom.addEventListener("click",HAniS.toggleZooming,false);
        divcon.appendChild(zoom);
        enableZooming = false;
      }

      if (cstr == "annotate") {
        annotate = make("button");
        configButton(annotate, "annotate", "Annotate", "Finished", mytip, "annotate");
        // create "divannot", but hide it
        divannot = make("div");
        divannot.setAttribute("style","position:absolute;box-sizing:content-box;height:120px;width:180px;top:20px;left:20px;z-index:15;visibility:hidden;background-color:lightgray;border-width:6px;border-style:solid;border-color:magenta;border-radius:25px;padding:10px;font:normal normal normal 16px Arial;color:black;");
        divannot.title="Drag this box around.  Select attributes, then go onto the image and click & drag";
        anxloc = 20;
        anyloc = 20;
        isAnnot = false;
        anyDown = false;
        anlastw = 3;
        anlastf = 11;
        anxdiff = anydiff = 0;
        ants = [];
        divannot.style["touch-action"] = "none";
        var reCoord = function(e) {
          anxloc = e.clientX - anxdiff;
          anyloc = e.clientY - anydiff;
          divannot.style.top = anyloc+"px";
          divannot.style.left = anxloc+"px";
        }

        // these 4 event handlers are for moving the div around
        divannot.addEventListener("mousedown", function(e) {
          if (e.clientX == 0 && e.clientY == 0) return;
          anyDown = true;
          anxdiff = e.clientX - anxloc;
          anydiff = e.clientY - anyloc;
        } , false);

        divannot.addEventListener("mouseup", function(e) {
          anyDown = false;
        } , false);

        divannot.addEventListener("mouseout", function(e) {
          if (!anyDown) return;
          reCoord(e);
        } , false);

        divannot.addEventListener("mousemove", function(e) {
          if (!anyDown) return;
          reCoord(e);
        }, false );

        if (window.PointerEvent) {
          divannot.addEventListener("pointerdown", function(e) {
            if (e.pointerType == "mouse") return;
            anxdiff = e.clientX - anxloc;
            anydiff = e.clientY - anyloc;
          }, false );

          divannot.addEventListener("pointermove", function(e) {
            if (e.pointerType == "mouse") return;
            if (anxdiff == 0) {
              anxdiff = e.clientX;
              anydiff = e.clientY;
            }
            reCoord(e);
          }, false );

        } else {
          divannot.addEventListener("touchmove", function(e) {
            //if (!anyDown) return;
            var et = e.changedTouches[0];
            if (anxdiff == 0) {
              anxdiff = et.clientX;
              anydiff = et.clientY;
            }
            reCoord(et);
          }, false );
        }


        annotate.addEventListener("click", function() {
          isAnnot = !isAnnot;
          if (isAnnot) {
            ants = [];
            anxloc = divall.offsetLeft;
            anyloc = divall.offsetTop;
            var al = configValues["annotate_position"];
            if (al != null) {
              var als = al.split(",");
              anxloc = anxloc + parseInt(als[0],10);
              anyloc = anyloc + parseInt(als[1],10);
            }
            divannot.style.top = anyloc+"px";
            divannot.style.left = anxloc+"px";
            divannot.style.visibility = "visible";
            toggleButton(annotate, false, "annotate off");
            anzoom = enableZooming;
            enableZooming = false;
            if (zoom != null) zoom.disabled = true;
          } else {
            divannot.style.visibility = "hidden";
            toggleButton(annotate, true, "annotate on");
            enableZooming = anzoom;
            if (zoom != null) zoom.disabled = false;
          }
          drawLines();
        }, true );

        ancol = makeSelect( ["Color","Black","White","Red","Green","Blue","Yellow","Orange","Cyan","Magenta"], ["black","black","white","red","green","blue","yellow","orange","cyan","magenta"], function(e) {
        });

        anwid = makeSelect( ["Width","1","2","3","4","5","6","7","8","10","12","14","16","18","20","24","28","32","36","38"],[1,1,2,3,4,5,6,7,8,10,12,14,16,18,20,24,28,32,36,38], function(e) {
          if (antype.selectedIndex == 6) {
            anlastf = anwid.selectedIndex;
          } else {
            anlastw = anwid.selectedIndex;
          }
        });
        anwid.title = "Select width or font size";

        antype = makeSelect( ["Type","Arrow","Circle","Box","Polygon","Draw","Text"],[-1,0,1,2,3,4,5], function(e) {
          if (antype.selectedIndex == 6) {
            antxt.focus();
            antxt.select();
            anwid.selectedIndex = anlastf;
          } else {
            anwid.selectedIndex = anlastw;
          }

          if (ancol.selectedIndex == 0) {
            ancol.selectedIndex = 2;
          }
        });

        var anera = make("button");
        anera.style.width="70px";
        anera.style.margin="5px";
        anera.innerHTML = "Erase";
        anera.title = "Erase last item";
        anera.addEventListener("click", function() {
          if (ants.length > 0) {
            ants.pop();
            drawLines();
          }
        }, false);

        antxt = make("input");
        antxt.type="text";
        antxt.style.width="150px";
        antxt.style.margin = "5px";
        antxt.title = "Enter your text here, then click where you want it";
        antxt.value="Text input";

        var cva = configValues["annotate_defaults"];
        if (cva != null) {
          // type, color, width, text
          var cvap = cva.split(",");
          antype.selectedIndex = parseInt(cvap[0],10);
          if (cvap.length > 1) ancol.selectedIndex = parseInt(cvap[1],10);
          if (cvap.length > 2) anwid.selectedIndex = parseInt(cvap[2],10);
          if (cvap.length > 3) antxt.value = cvap[3].trim();
        }

        var and0 = make("div");
        and0.innerHTML = "Annotation Tools";
        var and1 = make("div");
        and1.appendChild(antype);
        and1.appendChild(ancol);
        var and2 = make("div");
        and2.appendChild(anwid);
        and2.appendChild(anera);
        var and3 = make("div");
        and3.appendChild(antxt);

        divannot.appendChild(and0);
        divannot.appendChild(and1);
        divannot.appendChild(and2);
        divannot.appendChild(and3);
        divcon.appendChild(annotate);
        divall.appendChild(divannot);
      }

      if (cstr == "saveall") {
        saveall = make("button");
        configButton(saveall, "saveall", "SaveAll", "SaveAll", mytip, "saveall");
        saveallPrompt = configValues["saveall_prompt"];
        saveallFilename = configValues["saveall_filename"];
        var savl = configValues["saveall_list"];
        if (savl == null) {
          saveallList = ["0"];
        } else {
          saveallList = savl.split(",");
        }

        saveall.addEventListener("click", function() {
          for (var L=0; L<saveallList.length; L++) {
            var ln = parseInt(saveallList[L],10);
            for (var k=0; k<backFiles.length; k++) {
              if (k === 0) {
                if (saveallPrompt != null) {
                  saveallFilename = window.prompt(saveallPrompt, saveallFilename);
                  if (saveallFilename != null && saveallFilename.length < 1) {
                    saveallFilename=null;
                  }
                }
              }
              var fn;
              if (saveallFilename == null) {
                fn = "";
              } else {
                if (saveallFilename.indexOf("#") != -1) {
                  fn = saveallFilename.replace("#",k.toString());
                } else {
                  fn = saveallFilename+k;
                }
              }
              var alink = document.createElement("a");
              if (ln == 0) {
                alink.href = backFiles[k];
              } else {
                alink.href = overlayFiles[k][ln-1];
              }
              alink.download = fn;
              var evt = new MouseEvent("click");
              alink.dispatchEvent(evt);
            }
          }
        }, false);

        divcon.appendChild(saveall);
      }

      if (cstr == "capture") {
        capture = make("button");
        configButton(capture, "capture", "Capture", "Capture", mytip, "capture");
        var capturePrompt = configValues["capture_prompt"];
        captureFilename = configValues["capture_filename"];
        if (captureFilename == null) captureFilename = "haimage.png";
        capture.addEventListener("click", function() {
          var fn = captureFilename;
          if (capturePrompt != null) {
             fn = window.prompt(capturePrompt, fn);
             if (fn == null) return;
          }
          var savCan = make("canvas");
          savCan.height = imgCan.height;
          savCan.width = imgCan.width;
          var stx = savCan.getContext("2d");
          drawPrompts = false;
          drawLines();
          stx.drawImage(imgCan,0,0);
          stx.drawImage(drwCan,0,0);
          drawPrompts = true;
          savCan.toBlob( function(blob) {
            var image = URL.createObjectURL(blob);
            var alink = document.createElement("a");
            alink.href = image;
            alink.download = fn;
            var evt = new MouseEvent("click");
            alink.dispatchEvent(evt);
            setTimeout( function() {
              URL.revokeObjectURL(blob);
            }, 30000);

          });

        }, false);

        divcon.appendChild(capture);
      }

      if (cstr == "show") {
        show = make("button");
        configButton(show, "show", "Show", "Show", mytip, "show");
        showPrompt = configValues["show_prompt"];
        showOpt = configValues["show_image_file"];
        if (showPrompt == null) {
          showPrompt = "Right-click to save or copy the image<br> <font size='-1'>(to save in some browsers, you may need to first 'Open in New Tab')</font> <p>";
        }
        show.addEventListener("click", function() {
          if (showOpt) {
            var savwin = window.open(backFiles[curFrame]);
          } else {
            var candat = imgCan.toDataURL("image/png;base64;");
            var savwin = window.open();
            savwin.document.write(showPrompt+"<img src="+candat+"></img>");
          }

        }, false);

        divcon.appendChild(show);
      }

      if(cstr == "toggle" ) {
        toggleFrames = new Array();
        var togs = configValues["toggle_size"];
        if (togs != null) {
          var togv = togs.split(",");
          wTog = parseInt(togv[0],10);
          hTog = parseInt(togv[1],10);
          spTog = parseInt(togv[2],10);
        } else {
          wTog = 10;
          hTog = 10;
          spTog = 3;
        }
        togColorOff = "red";
        togColorOn = "blue";
        togColorSel = "orange";
        cwTog = false;
        if (beginsWith(configValues["toggle_layout_width"],"t")) cwTog = true;
        var togc = configValues["toggle_colors"];
        if (togc != null) {
          var togcs = togc.split(",");
          togColorOn = togcs[0].trim();
          togColorOff = togcs[1].trim();
          togColorSel = togcs[2].trim();
        }

        divtog = make("div");
        divtog.setAttribute("style",nosel);
        divtog.style.backgroundColor = divcon.style.backgroundColor;
        divtog.align=divalign;
        togPointer = new PEvs(divtog, null, null, null, null, HAniS.togclick, null);
        cantog = make("canvas");
        cantog.height = 2*hTog;
        ctxtog = cantog.getContext("2d");
        divtog.appendChild(cantog);
        if (isTop) {
          divall.insertBefore(divtog, divcan);
        } else {
          divall.appendChild(divtog);
        }
        var a = configValues["toggle_onoff"];
        toggleDefs = null;
        if (a != null) toggleDefs = a.split(",");

        useToggle = true;

        if (mytip != null) divtog.title = mytip

      }


      if (cstr == "overlay" || cstr == "menu") {

        overlayCheck = new Array();
        var olt = configValues["overlay_tooltip"];
        var oltips = null;
        if (olt != null) oltips = olt.split(",");
        var olr = configValues["overlay_radio"];
        var olrad = null;
        if (olr != null) {
          olrad = olr.split(",");
        }

        var olc = configValues["overlay_labels_color"];
        var olcolor = null;
        if (olc !=null) {
          olcolor = olc.split(",");
        }

        var oldrop = false;
        if (cstr == "menu") {
          oldrop = true;
        }


        overlayStatic = new Array();
        var ols = configValues["overlay_caching"];
        var olstat = null;
        if (ols == null) ols = configValues["overlay_static"];
        if (ols != null) olstat = ols.split(",");

        var oldv = true;
        if (configValues["overlay_nonewdiv"] != null) oldv = false;

        for (k=0; k<overlayLabels.length; k++) {
          var olab = overlayLabels[k].trim();

          if (k == 0 || olab.indexOf("/") == 0) {
            if (olab.indexOf("/") == 0) olab = olab.substr(1);
            if (oldrop) {
              divoldrop = make("div");
              divoldrop.setAttribute("style","text-align:left;display:block;visibility:hidden;position:absolute;border-style:solid;border-width:2px;background:#ffffff;z-index:999;"+configValues["overlay_labels_style"]);
              numdivolay++;

            } else if (oldv) {
              divolay = make("div");
              numdivolay++;
              divolay.align="center";
              if (isTop) {
                divall.insertBefore(divolay, divcan);
              } else {
                divall.appendChild(divolay);
              }
            } else {
              divolay = make("span");
              divcon.appendChild(divolay);
            }
            cv = configValues["overlay_labels_style"];
            if (!oldrop && cv != null) {
              divolay.setAttribute("style",cv);
            }
          }

          var olon = false;
          var oll = olab.length;
          if (olab.indexOf("/on") > 0 && olab.indexOf("/on") == oll-3) {
            olon =true;
            olab = olab.substr(0,oll-3);
          }

          var isAlways = false;
          if (olab.indexOf("/always")> 0 && olab.indexOf("/always")  == oll-7) {
            olon = true;
            isAlways = true;
          }

          var isHidden = false;
          if (olab.indexOf("/hidden") > 0) {
            isHidden = true;
          }

          var isIndex = false;
          if (olab.indexOf("/hotspots")> 0) {
            isIndex = true;
            olab = olab.substr(0, olab.indexOf("/"));
          }

          if (olstat != null) {
            if (beginsWith(olstat[k],"y") || beginsWith(olstat[k],"t")) {
              overlayStatic[k] = true;
            } else {
              overlayStatic[k] = false;
            }
          } else {
            overlayStatic[k] = true;
          }


          overlayCheck[k] = make("input");
          overlayCheck[k].type = "checkbox";
          overlayCheck[k].value = olab;
          overlayCheck[k].id = olab;
          overlayCheck[k].showMe = !(isAlways & isHidden) & !isIndex;

          var cbsty = "vertical-align:middle;";
          cv = configValues["checkbox_style"];
          if (cv != null) {
            cbsty = cbsty + cv;
          }
          overlayCheck[k].setAttribute("style", cbsty);
          overlayCheck[k].checked = olon;
          overlayCheck[k].restorableState = olon;
          overlayCheck[k].name = 0;
          overlayCheck[k].hotspotIndex = isIndex;
          if (enhance != null) {
            if ((k == overlayEnhNum) && overlayCheck[k].checked) {
              enhance.disabled = false;
            }
          }

          if (allowHoverzones != null && (!allowHoverzones[k] && olon)) okToShowHoverzones = false;
          overlayCheck[k].addEventListener("click",HAniS.overClick,false);
          if (olr != null) {
            var grp = olrad[k].indexOf("/");
            if (grp != -1) {
              overlayCheck[k].name = olrad[k].substr(grp+1).trim();
            }

            if (beginsWith(olrad[k],"t")) {
              overlayCheck[k].type="radio";
              if (grp == -1) {
                overlayCheck[k].name="all";
              }
            }
          }

          overlayCheck[k].isGhost = true;
          if (!isAlways && !isHidden) {
            if (oltips != null) overlayCheck[k].title = oltips[k];
            var lab = make('label');
            lab.htmlFor = olab;
            if (oltips != null) lab.title = oltips[k];
            lab.innerHTML = olab;
            lab.name = overlayCheck[k].name;
            var lsty = "vertical-align:middle;";
            if (olcolor != null) {
              lsty = lsty + "color:"+olcolor[k]+";";
            }

            lab.setAttribute("style",lsty);
            //if (!oldrop) lab.setAttribute("style",lsty);
            var spn = make("span");
            var space = "10"
            if (overlaySpacer != null) {
              space = 10 + overlaySpacer[k];
            }
            spn.appendChild(overlayCheck[k]);
            spn.appendChild(lab);
            if (oldrop) {
              spn.setAttribute("style","display:block;margin-right:5px;");
              divoldrop.appendChild(spn);
            } else {
              spn.setAttribute("style","margin-left:"+space+"px;");
              divolay.appendChild(spn);
            }
            overlayCheck[k].isGhost = false;
          }
        }

        if (oldrop) {
          var olbutt = make("button");
          configButton(olbutt,"menu","Show Overlay Menu","Hide Menu",mytip,"menu off");
          olbutt.addEventListener("click", function (e) {
            var vis = divoldrop.style.visibility;
            if (vis === "hidden") {
              divoldrop.style.visibility = "visible";
              toggleButton(olbutt, false, "menu on");
            } else {
              divoldrop.style.visibility = "hidden";
              toggleButton(olbutt, true, "menu off");
            }

          }, false);
          divcon.appendChild(olbutt);
          divall.insertBefore(divoldrop, divcan);
        }

        // now fix up links, if there
        if (overlayLinks == null) {
          overlayLinks = new Array(overlayLabels.length);
          for (k=0; k<overlayLabels.length; k++) {
            overlayLinks[k] = 0;
          }

        } else {
          for (k=0; k<overlayLabels.length; k++) {
            resetLinks(k);
          }
        }
      }
    }

    wasLooping = true;
    refLooping = null;
    cv = configValues["start_looping"];
    if (cv != null) {
      var a = cv.split(",");
      if (beginsWith(a[0], "f"))  wasLooping = false;
      if (a.length > 1) {
        begFrame = parseInt(a[1],10)-1;
        if (isNaN(begFrame)) begFrame = 0;
        curFrame = begFrame;
      }
      if (a.length > 2) {
        if (beginsWith(a[2], "t")) refLooping = true;
        if (beginsWith(a[2], "f")) refLooping = false;
      }
    }
    setIsLooping(wasLooping);

  }

  function resetLinks(i) {
    var k,state;

    if (overlayLinks[i] < 0) {
      state = false;
      for (k=0; k<overlayLabels.length; k++) {
        if (overlayLinks[k] == -overlayLinks[i]) {
          if (!state) state = overlayCheck[k].checked;
        }
      }
      overlayCheck[i].checked = state;
    }
  }


  function fetchHotspot(st) {

    var hro, hro2;
    if (numHotspots == 0) hotspots = new Array();
    hro = st.substr(st.indexOf("=")+1).trim();
    hro2 = hro.split(",");
    // hotspot=x,y,w,h,pan/olay#?frame#,action,value [,tip]
    // x,y,w,h,pan,action,value,overlay
    // x,y,icon,filename,pan/olay#?frame#, action, value
    // x,y,sprite,index,pan/olay#?frame#,...
    var pan = hro2[4].trim();
    var ps = hro2[4].indexOf("/");
    var ps2 = hro2[4].indexOf("?");
    var polay = -1;
    var pframe = -1;
    if (ps != -1) {
      polay = parseInt(hro2[4].substring(ps+1), 10)-1;
      pan = pan.substring(0,ps);
    }
    if (ps2 != -1) {
      pframe = parseInt(hro2[4].substring(ps2+1),10) -1;
      if (ps2 < ps || ps == -1) pan = pan.substring(0,ps2);
    }

    hotspots[numHotspots] = new Hotspot(
      parseInt(hro2[0].trim(),10), parseInt(hro2[1].trim(),10),
      hro2[2].trim(), hro2[3].trim(),
      pan.toLowerCase(),
      hro2[5].trim().toLowerCase(),
      hro2[6].trim(), polay, pframe,
      (hro2.length == 8 ? hro2[7].trim() : null)
    );

    numHotspots = numHotspots + 1;
  }

  function fetchHoverzone(st) {
    // Hoverzone = color, action, value, tip, poly
    // Hoverzone = 0xfe00fe, fof, temp/fof.txt, Click me, (100, 200, 110, ...)
    var hro, hro2;
    if (numHoverzones == 0) {
      hoverzones = new Array();
      hoverPick = null;
    }
    hro = st.substr(st.indexOf("=")+1).trim();
    var hroq = hro.split("(");
    hro2 = hroq[0].split(",");
    var col = hro2[0].trim();
    if (col.indexOf("0x") == 0) {
      col = "#"+col.substring(2);
    }

    hoverzones[numHoverzones] = new Hoverzone(
        col, hro2[1].trim(),
        hro2[2].trim(), hro2[3].trim(),hroq[1].trim()
    );
    numHoverzones = numHoverzones + 1;
  }

  function fetchHotzone(st) {
    var hro, hro2;
    if (numHotzones == 0) hotzones = new Array();
    hro = st.substr(st.indexOf("=")+1).trim();
    hro2 = hro.split(",");
    // hotzone = olay#, color, action, value [,tip]
    hotzones[numHotzones] = new Hotzone(
        parseInt(hro2[0],10)-1,
        parseInt(hro2[1],16),
        hro2[2].trim(), hro2[3].trim(),
        (hro2.length == 5 ? hro2[4].trim():"Click for details")
    );
    numHotzones = numHotzones + 1;
  }

  /** @constructor */
  function Hoverzone(color, action, value, tip, poly) {
    this.color = color;
    this.alpha = 1.0
    if (this.color.length > 7) {
      this.alpha = parseInt(this.color.substring(7),16) / 255.;
      this.color = this.color.substring(0,7);
      if (this.alpha < .1) this.alpha = .1;
    }
    this.action = action;
    this.value = value;
    this.tip = tip;
    this.poly = poly;
    var xyc = poly.split(",");
    this.xy = [];
    this.xmin = 9999;
    this.ymin = 9999;
    this.xmax = 0;
    this.ymax = 0
    for (var i=0; i<xyc.length; i++) {
      this.xy[i] = parseInt(xyc[i].trim(), 10)
      if (i % 2 == 0) {
        if (this.xy[i] < this.xmin) this.xmin = this.xy[i];
        if (this.xy[i] > this.xmax) this.xmax = this.xy[i];
      } else {
        if (this.xy[i] < this.ymin) this.ymin = this.xy[i];
        if (this.xy[i] > this.ymax) this.ymax = this.xy[i];
      }
    }
    this.height = this.ymax - this.ymin + 1;
    this.width = this.xmax - this.xmin + 1;

    this.can = make("canvas");
    this.can.height = this.height;
    this.can.width = this.width;
    this.ctx = this.can.getContext("2d");
    this.ctx.imageSmoothingEnabled=false;
    this.ctx.fillStyle = this.color;
    this.ctx.globalAlpha = this.alpha;

    this.ctx.beginPath();
    this.ctx.moveTo(this.xy[0]-this.xmin, this.xy[1]-this.ymin);
    for (var i=2; i<this.xy.length; i=i+2) {
      this.ctx.lineTo(this.xy[i]-this.xmin, this.xy[i+1]-this.ymin);
    }
    this.ctx.closePath();
    this.ctx.fill();
  }

  /** @constructor */
  function Hotzone(overlay, color, action, value, tip) {
    this.overlay = overlay;
    this.color = color;
    this.action = action;
    this.value = value;
    this.tip = tip;
  }


  /** @constructor */
  function Hotspot(x, y, w, h, pan, action, value, overlay, frame, tip) {
    this.x0= x;
    this.y0= y;
    this.icon = null;
    this.isIcon = false;
    if (w == "icon") {
      this.icon = new Image();
      this.icon.hotspot = this;
      isIconHotspot = true;
      this.icon.onload = function(eimg) {
        this.gotit = true;
        this.hotspot.x1 = this.hotspot.x0+this.width;
        this.hotspot.y1 = this.hotspot.y0+this.height;
        this.hotspot.width = this.width;
        this.hotspot.height = this.height;
        this.hotspot.w2 = this.width / 2;
        this.hotspot.h2 = this.height / 2;
        this.hotspot.isIcon = true;
        this.hotspot.origX = 0;
        this.hotspot.origY = 0;
      }

      this.x1 = -1;
      this.y1 = -1;
      this.anim = false;
      this.fromfile = false;
      this.zoomme = false;
      this.icon.gotit = false;
      this.icon.src = h.trim();

    } else if (w == "sprite" || w === "spritefn") {
      this.x0 = x;
      this.y0 = y;
      this.inx = parseInt(h,10) - 1;  // external starts at 1

      useSpriteFn = false;
      if (gotSprites) {
        if (spriteImages != null) {
          if (spriteImagesOffset != -1) {
            if (this.inx >= spriteImagesOffset && this.inx <= (spriteImagesOffset+spriteImages.length)) useSpriteFn = true;
          } else {
            if (w === "spritefn") useSpriteFn = true;
          }
        }
      } else {
        useSpriteFn = true;
      }

      if (useSpriteFn) {
        if (spriteImagesOffset != -1) this.inx = this.inx - spriteImagesOffset;
        if (this.inx >= spriteImages.length) return null;
        this.x1 = spriteImages[this.inx].width + x;
        this.y1 = spriteImages[this.inx].height + y;
        this.width = spriteImages[this.inx].width;
        this.height = spriteImages[this.inx].height;
        this.w2 = this.width/2;
        this.h2 = this.height/2;
        this.origX = 0;
        this.origY = 0;
        this.icon = spriteImages[this.inx].cloneNode();
        this.anim = spriteImages[this.inx].anim;
        this.fromfile = spriteImages[this.inx].fromfile;
        this.zoomme = spriteImages[this.inx].zoomme;
        if (this.anim) {
          this.icon.setAttribute("style","position:absolute;top:0px;left:0px;visibility:hidden;");
          divanim.appendChild(this.icon);
        }

      } else {
        this.x1 = spriteW[this.inx] + x;
        this.y1 = spriteH[this.inx] + y;
        this.width = spriteW[this.inx];
        this.height = spriteH[this.inx];
        this.w2 = spriteW[this.inx]/2;
        this.h2 = spriteH[this.inx]/2;
        this.origX = spriteX[this.inx];
        this.origY = spriteY[this.inx];
        this.icon = spriteImg;
        this.anim = false;
        this.fromfile = false;
        this.zoomme = false;
      }
      this.isIcon = true;
      isIconHotspot = true;
      this.icon.gotit = true;  // always assume

    } else {
      this.width = parseInt(w,10);
      this.height = parseInt(h,10);
      this.x1 = this.width+x;
      this.y1 = this.height+y;
      this.w2 = this.width/2;
      this.h2 = this.height/2;
    }

    this.pan = pan;
    this.isPan = false;
    if (this.pan === "pan") this.isPan = true;
    this.action = action;
    this.value = value;
    this.overlay = overlay;
    this.frame = frame;
    this.tip = tip;
  }

  this.drag = function(e) {

    if (isMark) return;

    xScreen = pointer.getX();
    yScreen = pointer.getY();

    if (isAnnot) {
      var anti = ants.length - 1;
      ants[anti].xe = xScreen;
      ants[anti].ye = yScreen;
      ants[anti].xp.push(xScreen);
      ants[anti].yp.push(yScreen);
      ants[anti].size = Math.round(Math.sqrt( Math.pow(xScreen - ants[anti].x,2) + Math.pow(yScreen - ants[anti].y,2)));
      drawLines();
      return;
    }

    xLoc = xScreen * canXScale;
    yLoc = yScreen * canYScale;

    showTip = false;

    showDistance = false;
    if (doDistance && ( !distShift || (distShift && e.shiftKey) ) ) {
      x1Dist = xLoc;
      y1Dist = yLoc;
      showDistance = true;
      drawLines();
      return;
    }

    if (enableZooming && (zoomYFactor != zoomYBase)) {

      xMove = xImage - xLoc/zoomXFactor;
      yMove = yImage - yLoc/zoomYFactor;
      doHide()
      xImage = Math.round(xMove + xLoc/zoomXFactor);
      yImage = Math.round(yMove + yLoc/zoomYFactor);

    } else if (!isMark && !isExtrap && doHoverzones) {
      hoverPick = null;
      if (okToShowHoverzones) {
        if (xLoc >= 0 && yLoc >= 0) {
          rgb = ctxh.getImageData(xLoc,yLoc,1,1).data;
          if (rgb[3] != 0) {
            hoverPick = hoverzones[rgb[3]-1];
          }
        }
      }
    }

    isDragging = true;
    drawIt();
    drawLines();
  }

  function setzoomFactorIndex() {
    if (!doingHiResZoom || doZoomFactors) return;
    zoomFactorIndex = -1;
    for (var i=0; i<zoomFactors.length; i++) {
      if (zoomXFactor >= zoomFactors[i]) {
        zoomFactorIndex = i;
      }
    }
  }

  function getLatLon() {
    if (locTran != null) {
      locll = locTran.toLatLon(xImage, yImage);
    } else if (loc0 != null) {
      locll[0] = loc0 + (loc2 - loc0)*yImage/imgHeight;
      locll[1] = loc1 + (loc3 - loc1)*xImage/imgWidth;
    } else {
      locll[0] = Math.round(xImage);
      locll[1] = Math.round(yImage);
    }
  }

  function doAction(item, type) {
    if (item.action == "popup") {
      if (popupWindow != null) popupWindow.close();
        popupWindow = window.open("","HAniSPopup","scrollbars=yes,width="+popupWinWidth+",height="+popupWinHeight);
        popupWindow.document.write(popupDiv+item.value+"</div>");

      } else if (item.action == "link") {
        var sitem = item.value;
        getLatLon();

        sitem = sitem.replace("%LATITUDE%",locll[0].toFixed(4));
        sitem = sitem.replace("%LONGITUDE%",locll[1].toFixed(4));
        sitem = sitem.replace("%FRAMENUMBER%",(curFrame+1));
        window.open(sitem,"_blank","");

      } else if (item.action == "fof") {
        imageBase = configImageBase;
        HAniS.resetZoom();
        HAniS.newFOF(item.value, true);
        clearOverlays(type);
      }
  }

  this.resetZoom = function() {

    zoomXFactor = zoomXBase;
    zoomYFactor = zoomYBase;
    zoomFactorIndex = -1;

    if (zoom != null) {
      toggleButton(zoom, true, "zoom on");
      enableZooming = false;
    }

    xImage = xLoc;
    yImage = yLoc;
    xMove = 0;
    yMove = 0;
    hImage = imgHeight;
    wImage = imgWidth;
    setzoomFactorIndex();
    drawIt();

    if (gotHoverzones) {
      doHoverzones = true;
    }
  }

  this.canTip = function() {
    var i;
    showTip = false;
    xScreen = pointer.getX();
    xLoc = xScreen * canXScale;
    yScreen = pointer.getY();
    yLoc = yScreen * canYScale;

    if (( isMark || isExtrap) && yLoc < extrapYend && yLoc > extrapYbeg && xLoc > extrapXbeg) {
      return;
    }

    xImage = Math.round(xMove + xLoc/zoomXFactor);
    yImage = Math.round(yMove + yLoc/zoomYFactor);

    if (numHotzones != 0) {
      for (i=0; i<numHotzones; i++) {
        if (hotzones[i].overlay >= numOverlays ) {
          info(" Error...hotzone overlay # "+hotzones[i].overlay+" > number of overlays = "+numOverlays);
          continue;
        }

        if (overlayCheck[hotzones[i].overlay].checked) {
          ctx1.clearRect(0,0,1,1);
          ctx1.drawImage(overlayImages[curFrame][hotzones[i].overlay],
                      xImage,yImage,1,1,0,0,1,1);
          rgb = ctx1.getImageData(0,0,1,1).data;
          rgbpack = (rgb[0]<<16) + (rgb[1]<<8) + rgb[2];

          if (rgbpack == hotzones[i].color) {
            showTip = true;
            tipX = xScreen;
            tipY = yScreen;
            tipText = hotzones[i].tip;
            break;
          }
        }
      }
    }

    if (numHotspots != 0) {
      var x,y, hit, hsi;
      for (i=numHotspots-1; i>=0; i--) {
        if (hotspots[i] == null) continue;
        hsi = hotspots[i];
        if (hsi.frame != -1 && hsi.frame != curFrame) continue;
        if (hsi.overlay != -1 &&
               !overlayCheck[hsi.overlay].checked) continue;

        hit = false;
        if (hsi.isPan) {
          if (hsi.isIcon) {
            x = (hsi.x0 - xMove)*zoomXFactor - hsi.w2;
            y = (hsi.y0 - yMove)*zoomYFactor - hsi.h2;
            if (xLoc > x && xLoc < x+hsi.width &&
                yLoc > y && yLoc < y+hsi.height) {
                  hit = true;
            }
          } else {
            x = (hsi.x0 - xMove)*zoomXFactor;
            y = (hsi.y0 - yMove)*zoomYFactor;
            if (xLoc > x && xLoc < x+hsi.width*zoomXFactor &&
                yLoc > y && yLoc < y+hsi.height*zoomYFactor) {
              hit = true;
            }
          }
        } else {
          if (xLoc > hsi.x0 && xLoc < hsi.x1 &&
                yLoc > hsi.y0 && yLoc < hsi.y1) {
             hit = true;
          }
        }

        if (hit && hsi.tip != null) {
          showTip = true;
          tipX = xScreen;
          tipY = yScreen;
          tipText = hotspots[i].tip;
          break;
        }
      }
    }
    drawLines();
  }

  this.move = function() {
    if (!isExtrap && doHoverzones) {
      hoverPick = null;
      if (okToShowHoverzones) {
        var x = pointer.getX() * canXScale;
        var y = pointer.getY() * canYScale;
        if (x >= 0 && y >= 0) {
          rgb = ctxh.getImageData(x,y,1,1).data;
          if (rgb[3] != 0) {
            hoverPick = hoverzones[rgb[3] - 1];
          }
        }
      }
    }
    if (isExtrap && extrapMode >= 2) {
      xScreen = pointer.getX();
      xLoc = xScreen * canXScale;
      yScreen = pointer.getY();
      yLoc = yScreen * canYScale;
    }
    drawLines();
  }


  this.down = function() {
    if (showTip) {
      showTip = false;
      drawLines();
    }

    xScreen = pointer.getX();
    xLoc = xScreen * canXScale;
    yScreen = pointer.getY();
    yLoc = yScreen * canYScale;

    x0Dist = xLoc;
    y0Dist = yLoc;
    x1Dist = x0Dist;
    y1Dist = y0Dist;
    xImage = Math.round(xMove + xLoc/zoomXFactor);
    yImage = Math.round(yMove + yLoc/zoomYFactor);
    isDragging = false;
    isDown = true;
    if (isAnnot) {
      var ant = {};
      ant.x = xScreen;
      ant.y = yScreen;
      ant.xe = xScreen;
      ant.ye = yScreen;
      ant.xp = [xScreen];
      ant.yp = [yScreen];
      ant.type = antype.value;
      ant.color = ancol.value;
      ant.width = anwid.value;
      ant.text = antxt.value;
      ant.size = 0;
      ants.push(ant);
    }
  }

  this.up = function(e) {
    showTip = false;
    isDown = false;

    if (isAnnot) {
      return;
    }

    if (isExtrap || isMark) {
      xScreen = pointer.getX();
      xLoc = xScreen * canXScale;
      yScreen = pointer.getY();
      yLoc = yScreen * canYScale;

      if (isMark) {
        markMode =+ 1;
        if (markMode > 0) {
          if (yLoc/canYScale < markHbeg) {
            if(xLoc/canXScale > markXbeg) {
              // delete last
              markPoints.pop();
              markPoints.pop();
              if (markPoints.length == 0) markMode = 0;
              drawLines();
              return;
            } else if (xLoc/canXScale < markXend) {
              drawLines();
              var markwin = window.open("","Mark Points");
              markwin.document.write("<div>Marked Points<br>");
              var marsav = [];
              for (var mp=0; mp<markPoints.length; mp=mp+2) {
                xImage = (xMove + markPoints[mp]/zoomXFactor);
                yImage = (yMove + markPoints[mp+1]/zoomYFactor);

                getLatLon();
                markwin.document.write(locll[0].toFixed(4)+","+locll[1].toFixed(4)+"<br>");
                if (mp == 0) {
                marsav[0] = locll[0];
                marsav[1] = locll[1];
                }
              }
              if (markClose) markwin.document.write(marsav[0].toFixed(4)+","+marsav[1].toFixed(4)+"<br>");
              return;
            }
          }
        }
        markPoints.push(xLoc);
        markPoints.push(yLoc);
        drawLines();
        return;
      }

      if (extrapMode < 3) {
        extrapX[extrapMode] = xLoc;
        extrapY[extrapMode] = yLoc;
        extrapT[extrapMode] = minutes[curFrame];
        if (extrapMode == 0) {
          extrapMode = 1;
          setCurrentFrame(findFrame(numFrames - 1));
          exMsg = 1;
          dirspdLabel = null;

        } else {
          dt = extrapT[1] - extrapT[0];
          dxdt = (extrapX[1] - extrapX[0]) / dt;
          dydt = (extrapY[1] - extrapY[0]) / dt;

          if (dirspdBox != null && distXScale != null) {
            var ddx = dxdt * distXScale;
            var ddy = dydt * distYScale;
            var speed = Math.round(Math.sqrt(ddx*ddx + ddy*ddy) * 60.);
            var dir = Math.atan2(ddx, -ddy)/.0174533;
            if (dir < 0.0) dir = dir + 360.;
            dirspdLabel = dirspdPrefix+compass[Math.round(dir/22.5)]+" at "+speed+" "+dirspdSuffix;

            dirspdX = extrapX[0];
            dirspdY = (dydt > 0) ? extrapY[0] - 15 : extrapY[0] + 30;
          }


          var timeFontSize = parseInt(timeFont,10);

          nmin= Math.round(Math.abs((timeFontSize+4)*4/dxdt));  // ap rox 14pt x 4 digits
          tmin= Math.round(Math.abs((timeFontSize+4)*2/dydt));  // ap rox 14pt x 2 height
          if (tmin < nmin) nmin = tmin;

          if (nmin > 120) {
            nmin = 120*Math.floor(nmin/120)+60;
          } else if (nmin > 90) {
            nmin = 120;
          } else if (nmin > 60) {
            nmin = 90;
          } else if (nmin > 30) {
            nmin = 60;
          } else if (nmin > 15) {
            nmin = 30;
          } else if (nmin > 10) {
            nmin = 15;
          } else if (nmin > 5) {
            nmin = 10;
          } else if (nmin > 2) {
            nmin = 5;
          } else {
            if (nmin == 0) nmin = 1;
          }

          tmin = (extrapT[1]/nmin*nmin) + nmin;

          exsign = 1;
          if (toFrom) exsign = -1;
          xInc = dxdt * nmin;
          yInc = dydt * nmin;

          setCurrentFrame(findFrame(numFrames - 1));
          extrapMode = 3;
          exMsg = 2;
        }

      } else if (yLoc/canYScale < extrapYend && yLoc/canYScale > extrapYbeg && xLoc/canXScale > extrapXbeg) {

        if (extrapMode == 3) {
          initExtrap();
          drawLines();
          return;
        }

      }
      drawLines();

    }

    if (showDistance) {
      showDistance = false;
      drawLines();
      return;

    } else {
      if (distance != null && enableZooming) {
        doDistance = false;
        toggleButton(distance, true, "distance on");
      }

    }

    if (isDragging) {
      isDragging = false;
      return;
    }

    if (e.altKey) {
      HAniS.resetZoom();
      return;
    }

    if (doHoverzones && hoverPick != null && showedHover) {
      doAction(hoverPick, "z");
      return;
    }

    xScreen = pointer.getX();
    xLoc = xScreen * canXScale;
    yScreen = pointer.getY();
    yLoc = yScreen * canYScale;

    if (numHotzones != 0) {
      for (var i=0; i<numHotzones; i++) {
        if (hotzones[i].overlay >= numOverlays ) {
          info(" Error...hotzone overlay # "+hotzones[i].overlay+" > number of overlays = "+numOverlays);
          continue;
        }

        if (overlayCheck[hotzones[i].overlay].checked) {
          ctx1.clearRect(0,0,1,1);
          ctx1.drawImage(overlayImages[curFrame][hotzones[i].overlay],Math.floor(xImage),Math.floor(yImage),1,1,0,0,1,1);
          rgb = ctx1.getImageData(0,0,1,1).data;
          rgbpack = (rgb[0]<<16) + (rgb[1]<<8) + rgb[2];

          if (rgbpack == hotzones[i].color) {
            doAction(hotzones[i], "z");
            return;
          }
        }
      }

    }

    if (numHotspots != 0) {
      var x,y, hit, hsi;
      for (var i=numHotspots-1; i>=0; i--) {
        if (hotspots[i] == null) continue;
        hsi = hotspots[i];
        if (hsi.frame != -1 && hsi.frame != curFrame) continue;
        if (hsi.fromfile) continue;
        if ((hsi.overlay != -1) && !overlayCheck[hsi.overlay].checked) continue;

        hit = false;
        if (hsi.isPan) {
          if (hsi.isIcon) {
            x = (hsi.x0 - xMove)*zoomXFactor - hsi.w2;
            y = (hsi.y0 - yMove)*zoomYFactor - hsi.h2;
            if (xLoc > x && xLoc < x+hsi.width &&
                yLoc > y && yLoc < y+hsi.height) {
                  hit = true;
            }
          } else {
            x = (hsi.x0 - xMove)*zoomXFactor;
            y = (hsi.y0 - yMove)*zoomYFactor;
            if (xLoc > x && xLoc < x+hsi.width*zoomXFactor &&
                yLoc > y && yLoc < y+hsi.height*zoomYFactor) {
              hit = true;
            }
          }
        } else {
          if (xLoc > hsi.x0 && xLoc < hsi.x1 &&
                yLoc > hsi.y0 && yLoc < hsi.y1) {
             hit = true;
          }
        }
        if (hit) {
           doAction(hotspots[i], "s");
           return;
        }
      }
    }

    if (enableZooming) {
      doHoverzones = false;
      doZoom( e.ctrlKey || isCtrlKey || (e.which && e.which == 3) || (e.button && e.button == 2))
    }

    drawIt();
    drawLines();
  }

  this.wheel = function(inout) {
    if (useWheelFrame) {
      incCurrentFrame(inout);
      return;
    }

    if (!enableZooming) return;
    xScreen = pointer.getX();
    xLoc = xScreen * canXScale;
    yScreen = pointer.getY();
    yLoc = yScreen * canYScale;
    xImage = Math.round(xMove + xLoc/zoomXFactor);
    yImage = Math.round(yMove + yLoc/zoomYFactor);
    doZoom( (inout == -1));
  }

  function doHide() {
    var sv = hideLeftZoom ? hideLeft/zoomXFactor : 0;
    if (xMove + sv < hideLeft) xMove = hideLeft - sv;

    sv = hideRightZoom ? hideRight/zoomXFactor : 0;
    if (xMove + wImage - sv > imgWidth - hideRight)  xMove = imgWidth - wImage + sv - hideRight;

    sv = hideTopZoom ? hideTop/zoomYFactor : 0;
      if (yMove + sv  < hideTop) yMove = hideTop - sv;

    sv = hideBottomZoom ? hideBottom/zoomYFactor : 0;
    if (yMove + hImage - sv > imgHeight-hideBottom)  yMove = imgHeight - hImage - hideBottom + sv;

    yMove = Math.round(yMove);
    xMove = Math.round(xMove);
  }

  function doZoom(goout) {

    var rz = false;
    if (goout) {
      if (doZoomFactors) {
        zoomFactorIndex --;
        if (zoomFactorIndex < 0) {
          rz = true;
        } else {
          zoomXFactor = zoomFactors[zoomFactorIndex];
          zoomYFactor = zoomFactors[zoomFactorIndex];
        }
      } else {
        zoomXFactor = zoomXFactor - 0.1 * zoomScale/canXScale;
        zoomYFactor = zoomYFactor - 0.1 * zoomScale/canYScale;
      }

    } else {
      if (doZoomFactors) {
        zoomFactorIndex++;
        if (zoomFactorIndex >= zoomFactors.length) {
          zoomFactorIndex --;
          if (cycleZoom) rz = true;
        } else {
          zoomXFactor = zoomFactors[zoomFactorIndex];
          zoomYFactor = zoomFactors[zoomFactorIndex];
        }
      } else {
        if (cycleZoom && (zoomXFactor === zoomFactorMax || zoomYFactor === zoomFactorMax)) rz = true;
        zoomXFactor = zoomXFactor + 0.1 * zoomScale/canXScale;
        zoomYFactor = zoomYFactor + 0.1 * zoomScale/canYScale;
        if (zoomXFactor > zoomFactorMax) zoomXFactor = zoomFactorMax;
        if (zoomYFactor > zoomFactorMax) zoomYFactor = zoomFactorMax;
      }
    }

    if (rz || zoomXFactor <= zoomXBase || zoomYFactor <= zoomYBase) {
      zoomXFactor = zoomXBase;
      zoomYFactor = zoomYBase;
      zoomFactorIndex = -1;
      xImage = xLoc;
      yImage = yLoc;
      xMove = 0;
      yMove = 0;
      hImage = imgHeight;
      wImage = imgWidth;
      hideBottom = 0;
      hideTop = 0;
      hideRight = 0;
      hideLeft = 0;
      if (gotHoverzones) doHoverzones = true;

    } else {
      hideBottom = hideBottomDef;
      hideTop = hideTopDef;
      hideRight = hideRightDef;
      hideLeft = hideLeftDef;
      xMove = xImage - xLoc/zoomXFactor;
      yMove = yImage - yLoc/zoomYFactor;
      hImage = Math.floor(imgHeight / zoomYFactor);
      wImage = Math.floor(imgWidth / zoomXFactor);
      doHide();
      if (gotHoverzones) doHoverzones = false;
    }
    setzoomFactorIndex();
  }

  this.toggleAutoRefresh = function() {
    if (isAutoRefresh) {
      toggleButton(autotoggle, false, "autorefresh off");
      if (refreshTimer != null) clearInterval(refreshTimer);
      isAutoRefresh = false;
    } else {
      toggleButton(autotoggle, true, "autorefresh on");
      refreshTimer = setInterval("HAniS.reloadFOF();",autoRefresh);
      isAutoRefresh = true;
    }
  }

  this.toggleLoopRock = function() {
    if (isRocking) {
      toggleButton(looprock, true, "looprock rock");
      isRocking = false;
    } else {
      toggleButton(looprock, false, "looprock loop");
      isRocking = true;
    }
  }

  this.toggleLoopHalt = function() {
    if (haltMe) {
      toggleButton(loophalt, true, "loophalt halt");
      haltMe = false;
      wasHalted = false;
    } else {
      toggleButton(loophalt, false, "loophalt loop");
      haltMe = true;
      wasHalted = true;
    }
  }

  this.toggleExtrap = function() {
    if (isExtrap) {
      if (extrap != null) {
        toggleButton(extrap, true, "extrap on");
        divall.style.cursor = prevCursor.pop();
        if (startstop != null) startstop.disabled = false;
        setIsLooping(wasLooping);
        enableZooming = wasZooming;
        extrapMode = -1;
      }
      isExtrap = false;
      drawLines();
    } else {
      if (extrap != null) {
        isExtrap = true;
        dirspdLabel = null;
        prevCursor.push(divall.style.cursor);
        divall.style.cursor = "crosshair";
        toggleButton(extrap, false, "extrap off");
        wasLooping = isLooping;
        if (startstop != null) startstop.disabled = true;
        isLooping = false;
        wasZooming = enableZooming;
        enableZooming = false;
        HAniS.resetZoom();
        initExtrap();
        drawLines();
      }
      drawIt();
    }
  }

  function initExtrap() {
    exMsg = 0;
    extrapX = new Array(2);
    extrapY = new Array(2);
    extrapT = new Array(2);
    extrapMode = 0;
    dirspdLabel = null;
    setCurrentFrame(findFrame(0));
  }

  function setIsLooping(r) {
    isLooping = !r;
    HAniS.toggleIsLooping();
  }

  function doAutoEnhance(img, ox, oy, ow, oh, tnum) {
    ctxae.clearRect(0,0,canW, canH);
    ctxae.imageSmoothingEnabled = enableSmoothing;
    ctx.imageSmoothingEnabled = enableSmoothing;
    ctxae.drawImage(img,ox,oy,ow,oh,0,0,canW, canH);
    ctxaed = ctxae.getImageData(0,0,canW,canH)
    eod = ctxaed.data;
    etr = tabR[tnum];
    etg = tabG[tnum];
    etb = tabB[tnum];
    eta = tabA[tnum];
    esd = 4 * canW * canH;
    for (ek=0; ek<esd; ek=ek+4) {
      eodk = eod[ek];
      if (eodk === eod[ek+1] && eodk === eod[ek+2]) {
        eod[ek] = etr[eodk];
        eod[ek+1] = etg[eodk];
        eod[ek+2] = etb[eodk];
        eod[ek+3] = eta[eodk];
      }
    }

    ctxae.putImageData(ctxaed,0,0);
    ctx.drawImage(aeCan,0,0,canW,canH);

  }

  this.doEnhance = function(e) {
    if (!isBaseEnh && !isOverlayEnh) return;
    var h = enhCan[0].height;
    var w = enhCan[0].width;
    var t = parseInt(enhance.value,10);
    var k,i,ed, od;
    if (t >= 0) {
      etr = tabR[t];
      etg = tabG[t];
      etb = tabB[t];
      eta = tabA[t];
      overlayProbe[overlayEnhNum] = t;
    } else {
      overlayProbe[overlayEnhNum] = null;

    }
    var sd = h * w * 4;
    for (i=0; i<origCan.length; i++) {
      if (enhCan[i].gotit) {
        enhCan[i].gotit = false;
        ed = enhID[i].data;
        od = origIDd[i];
        if (t < 0) {
          for (k=0; k<sd; k=k+4) {
            ed[k] = od[k]
            ed[k+1] = od[k+1];
            ed[k+2] = od[k+2];
            ed[k+3] = od[k+3];
          }
        } else {
          for (k=0; k<sd; k=k+4) {
            eodk = od[k];
            if (eodk === od[k+1] && eodk === od[k+2]) {
              ed[k] = etr[eodk];
              ed[k+1] = etg[eodk];
              ed[k+2] = etb[eodk];
              ed[k+3] = eta[eodk];
            }
          }
        }
        ctxe[i].putImageData(enhID[i],0,0);
        enhCan[i].gotit = true;
      }
    }
  }

  this.overClick = function(e) {
    var i;
    if (e.target.checked) {
      for (i=0; i<overlayCheck.length; i++) {
        if (e.target == overlayCheck[i]) continue;
        if (e.target.name != 0 && e.target.name == overlayCheck[i].name) {
          overlayCheck[i].checked = false;
        }
      }
    }

    okToShowHoverzones = true;
    for (i=0; i<overlayLinks.length; i++) {
      if (e.target != overlayCheck[i]) {
        resetLinks(i);
      }
      if (allowHoverzones != null &&
        (!allowHoverzones[i] && overlayCheck[i].checked)) okToShowHoverzones = false;

      if (enhance != null) {
        if (i === overlayEnhNum) {
          enhance.disabled = !overlayCheck[i].checked;
        }
      }
    }
    drawIt();
  }

  function toggleButton(butt, on, cn) {
    if (butt == null) return;
    if (on) {
      if (butt.style_on != null) butt.setAttribute("style",butt.style_on);
      butt.innerHTML = butt.label_on;
    } else {
      if (butt.style_off != null) butt.setAttribute("style",butt.style_off);
      butt.innerHTML = butt.label_off;
    }
    if (useCN) butt.className = cn;
  }

  this.toggleIsLooping = function() {
    if (isLooping) {
      isLooping = false;
      toggleButton(startstop, true, "startstop start");
    } else {
      isLooping = true;
      toggleButton(startstop, false, "startstop stop");
      wasHalted = haltMe;
      haltMe = false;
    }
  }


  this.toggleZooming = function() {
    if (enableZooming) {
      HAniS.resetZoom();
      toggleButton(zoom, true, "zoom on");
    } else {
      toggleButton(zoom, false, "zoom off");
      enableZooming = true;
      if (distance != null) {
        doDistance = false;
        toggleButton(distance, true, "distance on");
      }
      if (gotHoverzones) doHoverzones = true;
      if (isExtrap) HAniS.toggleExtrap();
    }
  }

  function setToggleState(n,s) {
    if (n >= numFrames) return;
    if (toggleFrames[n] == -2) s=-2;
    toggleFrames[n] = s;
    if (isSetframe && s == 1) {
      setframe.value = n+1;
      setframeLabelSpan.innerHTML = setframeLabel.replace("*",(n+1));
    }
    if (!useToggle) return;
    var x = togstart + n*(wTog + spTog);
    var c = togColorSel;
    if (s == 0) {
      c = togColorOn;
    } else if (s == -1) {
      c = togColorOff;
    } else if (s == -2) {
      c = missTogColor
    }
    ctxtog.fillStyle = c;
    ctxtog.fillRect(x,hTog/2, wTog, hTog);
  }

  this.togclick = function(e) {
    var x = togPointer.getX();
    var y = togPointer.getY();
    var xf = togstart;
    for (var i=0; i<toggleFrames.length; i++) {
      if (x > xf && x<xf + wTog) {
        if (e.shiftKey) {
          if (toggleFrames[i] >= 0) {
            setIsLooping(false);
            setCurrentFrame(i);
          }
          break;
        }
        var s = 0;
        if (toggleFrames[i] >= 0) s = -1;
        setToggleState(i, s);
        break;
      }
      xf = xf + wTog + spTog;
    }
  }

  function makeToggles(n, init) {
    if (useToggle) {
      if (cwTog) {
        cantog.width = parseInt(divall.style.width,10);
      } else {
        cantog.width = canW;
      }
      ctxtog.clearRect(0,0,cantog.width, cantog.height);
      togstart = cantog.width/2 - n*(wTog+spTog)/2;
    }
    for (var i=0; i<n; i++) {
      if (init) {
        if (missTog != -1) {
          setToggleState(i,-2);
        } else if (toggleDefs != null && toggleDefs[i].toLowerCase().trim()=="n") {
          setToggleState(i,-1);
        } else {
          setToggleState(i,0);
        }
      } else {
        setToggleState(i, toggleFrames[i]);
      }
    }

  }

  this.doneLoading = function(callback) {
    cdLoading = callback;
  }

  this.dwellChanged = function(callback) {
    dwChanged = callback;
  }

  this.getDwell = function() {
    return dwell;
  }

  this.frameChanged = function(callback) {
    cfChanged = callback;
  }

  this.getFrame = function() {
    return curFrame;
  }

  this.showFrame = function(f) {
     setCurrentFrame(findFrame(f));
  }

  function setCurrentFrame(n) {
    setToggleState(curFrame,0);
    curFrame = n;
    setToggleState(curFrame,1);
    if (frameLabelField != null && frameLabels != null &&
                               frameLabels[curFrame] != null) {
      frameLabelField.innerHTML = frameLabels[curFrame];
    }
    if (cfChanged != undefined) cfChanged(curFrame);
  }

  function findFrame(f) {
    if (f >= numFrames) f = numFrames - 1;
    if (toggleFrames[f] >= 0) return f;
    var d = -1, nf = f;
    if (nf == 0) d = 1;
    for (var i=0; i<numFrames-1; i++) {
      nf = nf + d;
      if (toggleFrames[nf] >= 0) return nf;
    }
    return f;
  }

  function incCurrentFrame(direct) {
    setToggleState(curFrame,0);
    var n = direct;
    do {
      if (n > 0) {
        var topf = findFrame(numFrames-1);
        curFrame = curFrame + 1;
        if (curFrame > topf) {
          if (isRocking && isLooping) {
            curFrame = topf;
            direct = -direct;
            if (haltMe) setIsLooping(false);
          } else if (haltMe) {
            curFrame = topf;
            setIsLooping(false);
          } else {
            curFrame = findFrame(0);
          }
        }

      } else {
        var botf = findFrame(0);
        curFrame = curFrame - 1;
        if (curFrame < botf) {
          if (isRocking && isLooping) {
            curFrame = botf;
            direct = -direct;
            if (haltMe) setIsLooping(false);
          } else if (haltMe) {
            curFrame = botf;
            setIsLooping(false);
          } else {
            curFrame = findFrame(numFrames - 1);
          }
        }
      }

      n = direct;
    } while (toggleFrames[curFrame] < 0);

    haltMe = wasHalted;


    if (cfChanged != undefined) cfChanged(curFrame);
    direction = n;
    setToggleState(curFrame,1);
    if (frameLabelField != null && frameLabels != null &&
                               frameLabels[curFrame] != null) {
      frameLabelField.innerHTML = frameLabels[curFrame];
    }

  }

  /** @constructor */
  function TextBox(bg, fg, font, scolor, sblur, sx, sy) {
    this.bg = bg;
    if (bg != null && this.bg.indexOf("0x") == 0) {
      this.bg = "#"+this.bg.substring(2);
    }
    this.fg = fg;
    if (fg != null && this.fg.indexOf("0x") == 0) {
      this.fg = "#"+this.fg.substring(2);
    }
    this.font = font;
    this.fontHeight = Math.round(1.1 * parseInt(font,10));
    this.scolor = scolor;
    if (scolor != null && this.scolor.indexOf("0x") == 0) {
      this.scolor = "#"+this.scolor.substring(2);
    }
    this.sblur = sblur;
    this.sxoff = sx;
    this.syoff = sy;
  }

  function drawText(box, x, y, text, lcr) {
    ctxd.globalAlpha = 1.0;
    ctxd.font = box.font;
    var fh = box.fontHeight;
    var yp = y - fh - 6;
    var zp = ctxd.measureText(text).width;
    var xp = x;
    if (lcr == 1) xp = xp - zp;
    if (lcr == 0) xp = xp - zp/2;

    if (xp < 5) xp = 5;
    if (yp < 5) yp = 5;
    if (yp + fh + 3 > canH)
             yp = canH - fh - 6;
    if (xp + 3 + zp > canW) xp = canW - zp - 6;

    ctxd.save();
    ctxd.beginPath();
    ctxd.fillStyle = box.bg;
    if (box.scolor != null) {
      ctxd.shadowColor = box.scolor;
      ctxd.shadowBlur = box.sblur;
      ctxd.shadowOffsetX = box.sxoff;
      ctxd.shadowOffsetY = box.syoff;
    }
    xText = xp - 3;
    yText = yp;
    wText = zp + 6;
    hText = fh + 6;

    ctxd.fillRect(xText,  yText, wText, hText);
    ctxd.closePath();
    ctxd.restore();

    ctxd.textBaseline = "bottom";
    ctxd.fillStyle = box.fg;
    ctxd.fillText(text, xp, yp + fh + 3);
  }

  function drawLines() {

    ctxd.clearRect(0,0,canW,canH);
    showedHover = false;

    if (isExtrap) {
      var pp = 15;
      ctxd.beginPath();
      ctxd.lineWidth = 2;
      ctxd.strokeStyle = "white";
      var ir = extrapMode;
      if (ir > 1) ir = 2;
      for (var i=0; i<ir; i++) {
        if (extrapMode > 1) {
          ctxd.moveTo(extrapX[0]/canXScale, extrapY[0]/canYScale);
          ctxd.lineTo(extrapX[1]/canXScale, extrapY[1]/canYScale);
        }
        ctxd.moveTo(extrapX[i]/canXScale - pp, extrapY[i]/canYScale);
        ctxd.lineTo(extrapX[i]/canXScale + pp, extrapY[i]/canYScale);
        ctxd.moveTo(extrapX[i]/canXScale, extrapY[i]/canYScale - pp);
        ctxd.lineTo(extrapX[i]/canXScale, extrapY[i]/canYScale + pp);
      }

      ctxd.closePath();
      ctxd.stroke();

      if (extrapMode == 3) {

        var x = xScreen;
        var y = yScreen;

        ctxd.moveTo(x,y);
        ctxd.fillStyle = timeColor;

        ctxd.fillRect(x-2, y-2, 5, 5);

        var accumTime = tmin;
        var endX = Math.round(x + exsign*(dxdt/canXScale*(tmin - extrapT[1])));
        var endY = Math.round(y + exsign*(dydt/canYScale*(tmin - extrapT[1])));

         // fillRect(x-2,y-2,5,5)

        for (var gasp=0; gasp < 300; gasp++) {

           ctxd.beginPath();
           ctxd.lineWidth = 1;
           ctxd.strokeStyle = timeColor;
           ctxd.moveTo(x,y);
           ctxd.lineTo(endX, endY);
           ctxd.stroke();
           ctxd.beginPath();
           ctxd.lineWidth = 3;
           ctxd.rect(endX-2, endY-2, 5,5);
           ctxd.stroke();

           if (endX < 3 || (endX+3) > canW ||
               endY < 3 || (endY+3) > canH) break;

           var hm = startingMinute + accumTime;
           hm = 100*Math.floor(hm/60) + (hm % 60);
           hm = hm % 2400;

           var shm;
           if (extrapAMPM) {

             var hampm= "AM";
             if (hm >= 1200) {
               if (hm >= 1300) hm = hm - 1200;
               hampm = "PM";
             } else if (hm < 100) {
               hm = hm + 1200;
             }
             shm = Math.floor(hm/100) +":"+ Math.floor(Math.floor(hm % 100)/10) + Math.floor(hm % 10) + " "+hampm;

           } else {
             shm = hm+" ";
             if (hm < 1000) shm = "0"+hm;
             if (hm < 100) shm = "00"+hm;
             if (hm < 10) shm = "000"+hm;
           }

           // here we draw the time label.....

           var rot = 0.0;
           if (Math.abs(yInc) < (timeFontSize+5)) {
             rot = 45.*.0174533;
           }

           var yb = endY;
           if (xInc * yInc < 0 && rot == 0) {
             yb = endY + timeFontSize;
           }
           ctxd.font = timeFont;
           shm = shm + " " + tzLabel;
           var zp = ctxd.measureText(shm).width;
           ctxd.fillStyle = timeBack;

           if (rot === 0.0) {
             ctxd.fillRect(endX+2, yb - timeFontSize-2 + rot, zp+4, timeFontSize+3);
             ctxd.fillStyle = timeColor;
             ctxd.textBaseline = "bottom";
             ctxd.fillText(shm, endX+5, yb);
           } else {

             ctxd.save();
             ctxd.translate(endX+5,endY);
             ctxd.rotate(rot);
             ctxd.fillRect(2, -2, zp+4, timeFontSize+3);
             ctxd.fillStyle = timeColor;
             ctxd.textBaseline = "bottom";
             ctxd.fillText(shm, 5, timeFontSize);
             ctxd.restore();
           }

           x = endX;
           y = endY;
           endX = Math.round(x + exsign*xInc);
           endY = Math.round(y + exsign*yInc);
           accumTime = accumTime + nmin;
         }

       }

       if (dirspdLabel != null) {
         drawText(dirspdBox, dirspdX, dirspdY, dirspdLabel, 0);
       }
       if (drawPrompts) {
         drawText(extrapTB, 9999, extrapYpos, extrapPrompts[exMsg], 0);
         extrapXbeg = xText;
         extrapYbeg = yText;
         extrapYend = yText + hText;
       }
    }

    if (isMark) {
      if (drawPrompts) {
        if (markMode == 0) {
          drawText(markTB, 9999, 10, markPrompts[0], 0);
          markXbeg = xText;
          markHbeg = hText;
        } else {
          drawText(markSaveTB, 10, 10, markPrompts[1], 0);
          markXend = xText+wText;
          drawText(markTB, 9999, 10, markPrompts[2], 0);
          markXbeg = xText;
          markHbeg = hText;
        }
      }

      ctxd.beginPath();
      ctxd.lineWidth = 2;
      ctxd.strokeStyle = markColor;
      var mps;
      for (mps=0; mps<markPoints.length; mps=mps+2) {
        if (mps == 0) {
          ctxd.moveTo(markPoints[mps]/canXScale, markPoints[mps+1]/canYScale);
        } else {
          ctxd.lineTo(markPoints[mps]/canXScale, markPoints[mps+1]/canYScale);
        }
      }
      if (markClose & markPoints.length>4) ctxd.lineTo(markPoints[0]/canXScale, markPoints[1]/canYScale);
      ctxd.stroke();
      for (mps=0; mps<markPoints.length; mps=mps+2) {
        ctxd.beginPath();
        ctxd.rect(markPoints[mps]/canXScale-2, markPoints[mps+1]/canYScale-2, 5,5);
        ctxd.stroke();
      }
    }

    if (isAnnot) {
      for (var ia = 0; ia<ants.length; ia++) {
        var at = ants[ia].type;
        ctxd.beginPath();
        ctxd.strokeStyle = ants[ia].color;
        ctxd.lineWidth = ants[ia].width;
        if (at == "0") { // arrow
          var ang = Math.atan2( (ants[ia].xe-ants[ia].x), (ants[ia].ye-ants[ia].y));
          var xn = ants[ia].x + ants[ia].width*Math.sin(ang);
          var yn = ants[ia].y + ants[ia].width*Math.cos(ang);
          ctxd.beginPath();
          ctxd.moveTo(xn,yn);
          ctxd.lineTo(ants[ia].xe, ants[ia].ye);
          var heads = 12+ants[ia].width/5;
          var heada = 30*.0174533;
          ctxd.moveTo( xn+heads*Math.sin(ang-heada), yn+heads*Math.cos(ang-heada));
          ctxd.lineTo(xn,yn);
          ctxd.lineTo( xn+heads*Math.sin(ang+heada), yn+heads*Math.cos(ang+heada));

        } else if (at == "1") {  // circle
          ctxd.arc(ants[ia].x, ants[ia].y, ants[ia].size,0.,2.0*Math.PI);

        } else if (at == "2") { // box
          ctxd.rect( ants[ia].x - ants[ia].size, ants[ia].y - ants[ia].size, 2*ants[ia].size, 2*ants[ia].size);
        } else if (at == "3") { // polygon
          ctxd.moveTo(ants[ia].xp[0], ants[ia].yp[0]);
          for (var xn=1;xn<ants[ia].xp.length;xn++) {
            ctxd.lineTo(ants[ia].xp[xn], ants[ia].yp[xn]);
          }
          ctxd.lineTo(ants[ia].xp[0], ants[ia].yp[0]);

        } else if (at == "4") { // draw
          ctxd.moveTo(ants[ia].xp[0], ants[ia].yp[0]);
          for (var xn=1;xn<ants[ia].xp.length;xn++) {
            ctxd.lineTo(ants[ia].xp[xn], ants[ia].yp[xn]);
          }

        } else if (at == "5") { // text
          ctxd.fillStyle = ants[ia].color;
          ctxd.font = ants[ia].width+"pt arial";
          var anz = ctxd.measureText(ants[ia].text).width;
          ctxd.fillText(ants[ia].text, ants[ia].x-anz/2, ants[ia].y+ants[ia].width/2);
        }
        ctxd.stroke();
      }

    }

    if (showDistance) {

      var dx = distXScale *(x0Dist - x1Dist)/zoomXFactor;
      var dy = distYScale *(y0Dist - y1Dist)/zoomYFactor;
      var distVal = " "+(Math.sqrt(dx*dx + dy*dy)).toFixed(distDigits)+" "+distUnit+" ";

      drawText(distBox, x1Dist/canXScale, y1Dist/canYScale, distVal, -1);

      ctxd.beginPath();
      ctxd.strokeStyle = distLineColor;
      ctxd.lineWidth = 3;
      ctxd.moveTo(x0Dist/canXScale, y0Dist/canYScale);
      ctxd.lineTo(x1Dist/canXScale, y1Dist/canYScale);
      ctxd.stroke();
      ctxd.closePath();
    }

    if (showTip) {
      drawText(tipBox, tipX, tipY, tipText, -1);
    }

    if (showProbe || showLocation) {
      xScreen = pointer.getX();
      xLoc = xScreen * canXScale;
      yScreen = pointer.getY();
      yLoc = yScreen * canYScale;

      if (xLoc >= 0 && yLoc >= 0) {
        xImage = (xMove + xLoc/zoomXFactor);
        yImage = (yMove + yLoc/zoomYFactor);

        if (showLocation) {
          getLatLon();
          llstr = locLatPrefix + locll[0].toFixed(locDigits)+"  "+ locLonPrefix+locll[1].toFixed(locDigits);
          var ypos = yScreen;
          if (showDistance || showTip) ypos = yScreen + locBox.fontHeight+6;
          drawText(locBox, xScreen+10, ypos, llstr, -1);
        }

        if (showProbe && gotTable && gotImages) {
          xImage = Math.round(xImage);
          yImage = Math.round(yImage);
          if (doBaseProbe) {
            ctx1.clearRect(0,0,1,1);
            ctx1.drawImage(backImages[curFrame],xImage,yImage,1,1,0,0,1,1);
            rgb = ctx1.getImageData(0,0,1,1).data;
            tn = 0;
            probeScale(false);

          } else {
            var mk,k
            for (mk=0; mk<numOverlays; mk++) {
              k = numOverlays - mk - 1;
              if (overlayImages[curFrame][k].gotit &&
                  overlayCheck[k].checked && (k == overlayEnhNum ||
                  (overlayProbe[k] != null && overlayProbe[k] >= 0))) {

                ctx1.clearRect(0,0,1,1);
                if (k == overlayEnhNum) {
                  ctx1.drawImage(origCan[curFrame],xImage,yImage,1,1,0,0,1,1);
                } else {
                  ctx1.drawImage(overlayImages[curFrame][k],xImage,yImage,1,1,0,0,1,1);
                }
                rgb = ctx1.getImageData(0,0,1,1).data;
                tn = overlayProbe[k];
                if (tn != null) probeScale(( k == overlayEnhNum) || tabEnh);

                break;
              }
            }
          }
        }
      }
    }

    if (doHoverzones && !isExtrap && !doDistance && !showProbe) {
      if (hoverPick == null) return;
      ctxd.drawImage(hoverPick.can, 0, 0, hoverPick.width,
      hoverPick.height, hoverPick.xmin/canXScale,
      hoverPick.ymin/canYScale, hoverPick.width/canXScale,
      hoverPick.height/canYScale);

      var xx = (hoverPick.xmin + hoverPick.xmax)/2
      var yy = (hoverPick.ymin + hoverPick.ymax)/2
      drawText(tipBox, xx/canXScale, yy/canYScale, hoverPick.tip, 0);
      showedHover = true;
    }

  }

  function probeScale(pt) {
    minDiff = 999;
    pValue = probeUndef;
    var tvl = tabVal[tn].length - 1;

    if (rgb[3] == 0) {
      pValue = tabMissing[tn];
    } else {

      if (probeExact) {
        diffInx = rgb[0];
      } else if (pt) {
        diffPct = 0.0;
        diffInx = rgb[0];
        minDiff = 0;
      } else {

        for (var i=0; i<tvl; i++) {

          drgb[0] = (rgb[0] - tabR[tn][i]);
          if (drgb[0]*(tabR[tn][i+1] - rgb[0]-1) < 0) continue;

          drgb[1] = (rgb[1] - tabG[tn][i]);
          if (drgb[1]*(tabG[tn][i+1] - rgb[1]-1) < 0) continue;

          drgb[2] = (rgb[2] - tabB[tn][i])
          if (drgb[2]*(tabB[tn][i+1] - rgb[2]-1) < 0) continue;

          minx = tabInx[tn][i];
          pct = drgb[minx]/tabDif[tn][i][minx];

          diff =
           Math.abs(drgb[m1[minx]] - pct*tabDif[tn][i][m1[minx]])+
           Math.abs(drgb[m2[minx]] - pct*tabDif[tn][i][m2[minx]]);

           if (diff < minDiff) {
             diffInx = i;
             minDiff = diff;
             diffPct = pct;
           }

        }
      }

      if (minDiff < 999) {
        pValue = tabPrefix[tn][diffInx];
        if (tabDecimal[tn][diffInx] != -1) {
          dbzz = tabVal[tn][diffInx];
          if (diffPct != 0) dbzz = dbzz + diffPct*(tabVal[tn][diffInx+1] - tabVal[tn][diffInx]);
          pValue = pValue+" "+ dbzz.toFixed(tabDecimal[tn][diffInx])+
                  " "+tabUnit[tn][diffInx];
        }

      }  else if (pt || (probeTest > 0)) {
        var j = -1;
        if ( pt ) {
          j = rgb[0]
        } else {
          for (var i=0; i<tvl; i++) {
            if (Math.abs(rgb[0] - tabR[tn][i]) < probeTest &&
                Math.abs(rgb[1] - tabG[tn][i]) < probeTest &&
                Math.abs(rgb[2] - tabB[tn][i]) < probeTest) {
              j = i;
              break;
            }
          }
        }
        if (j != -1) {
            pValue = tabPrefix[tn][j];
            if (tabDecimal[tn][j] != -1) {
              pValue = pValue+" "+
                tabVal[tn][j].toFixed(tabDecimal[tn][j])+ " "+
                   tabUnit[tn][j];
            }
        }
      }
    }
    if (pValue != "hide") drawText(probeBox, xScreen-10, yScreen, pValue, 1);
  }

  function drawIt() {
    var i;
    if (!gotImages) return;
    try {
      ctx.clearRect(0,0,canW, canH);

      ctx.globalAlpha = 1.0;

      if (!hideBackground) {

        if (doingHiResZoom && hiResBase != null && zoomFactorIndex >= 0 && hiResBase[zoomFactorIndex].gotit) {
           ctx.drawImage(hiResBase[zoomFactorIndex],xMove*zoomFactors[zoomFactorIndex],yMove*zoomFactors[zoomFactorIndex],wImage*zoomFactors[zoomFactorIndex],hImage*zoomFactors[zoomFactorIndex],0,0,canW, canH);

        } else {
          if (backImages[curFrame].gotit) {
             if (autoEnhanceBg != null) {
               doAutoEnhance(backImages[curFrame],xMove,yMove,wImage,hImage,autoEnhanceBg);
             } else {
               ctx.drawImage(backImages[curFrame],xMove,yMove,wImage,hImage,0,0,canW, canH);
             }
           }

          if (preserveBackPoints != null) {
            ctx.globalAlpha = 1.0;
            if (backImages[curFrame].gotit) {
              for (var ii = 0; ii < preserveBackPoints.length; ii=ii+4) {
                ctx.drawImage(backImages[curFrame],
                  preserveBackPoints[ii], preserveBackPoints[ii+1],
                    preserveBackPoints[ii+2], preserveBackPoints[ii+3],
                    preserveBackPoints[ii]/canXScale,
                    preserveBackPoints[ii+1]/canYScale,
                    preserveBackPoints[ii+2]/canXScale,
                    preserveBackPoints[ii+3]/canYScale);
              }
            }
          }

        }
      }

      if (numOverlays > 0) {
        for (var ii=0; ii<numOverlays; ii++) {
          i = (overlayOrder != null) ? overlayOrder[ii] : ii;
          if (overlayTop[i]) continue;
          if (overlayCheck[i].checked && overlayCheck[i].showMe) {
            if (overlayAlpha != null) ctx.globalAlpha = overlayAlpha[i];

            if (doingHiResZoom && zoomFactorIndex >= 0 && hiResOlay != null &&
            hiResOlay[i] != undefined && hiResOlay[i][zoomFactorIndex][curFrame].gotit) {
                ctx.drawImage(hiResOlay[i][zoomFactorIndex][curFrame],xMove*zoomFactors[zoomFactorIndex],yMove*zoomFactors[zoomFactorIndex],wImage*zoomFactors[zoomFactorIndex],hImage*zoomFactors[zoomFactorIndex],0,0,canW, canH);

            } else {
              if (overlayImages[curFrame][i].gotit) {
                if (olayZoomIndex == null || olayZoomIndex[i] === 1) {
                  if (autoEnhanceList != null && autoEnhanceList[i] >= 0) {
                    doAutoEnhance(overlayImages[curFrame][i],xMove,yMove,wImage,hImage, autoEnhanceList[i]);

                  } else {
                    ctx.drawImage(overlayImages[curFrame][i],xMove,yMove,wImage,hImage,0,0,canW, canH);
                  }
                } else if (olayZoomIndex[i] === 0) {
                    ctx.drawImage(overlayImages[curFrame][i],0,0, imgWidth, imgHeight, 0, 0, canW, canH);
                }
              }

            }
          }
        }

        if (preserveIndex != null) {
          ctx.globalAlpha = 1.0;
          for (var ii=0; ii<numOverlays; ii++) {
            i = (overlayOrder != null) ? overlayOrder[ii] : ii;
            if (overlayImages[curFrame][i].gotit &&
                  (preserveAlways[i] || overlayCheck[i].checked) && preserveIndex[i]) {
              for (var kp =0; kp < preservePoints[i].length; kp++) {
                ctx.drawImage(overlayImages[curFrame][i].origImage != null ?
                    overlayImages[curFrame][i].origImage : overlayImages[curFrame][i],
                preservePoints[i][kp][0], preservePoints[i][kp][1],
                preservePoints[i][kp][2], preservePoints[i][kp][3],
                preservePoints[i][kp][4]/canXScale, preservePoints[i][kp][5]/canYScale,
                preservePoints[i][kp][2]/canXScale, preservePoints[i][kp][3]/canYScale);
              }
            }
          }
        }

        if (showProbe) drawLines();
      }

      if (useProgress) drawImageProgress();

      if (isIconHotspot) {
        ctx.globalAlpha = 1.0;
        var hsi, xp, yp;
        for (i=0; i<numHotspots; i++) {
          if (hotspots[i] == null) continue;
          hsi = hotspots[i];
          if (hsi.icon != null) hsi.icon.style.visibility = "hidden";
          if (hsi.frame != -1 && hsi.frame != curFrame) continue;
          if ((hsi.icon != null) && hsi.icon.gotit &&
                 (hsi.overlay == -1 ||
                 (hsi.overlay != -1 && overlayCheck[hsi.overlay].checked))) {
            if (hsi.isPan) {
              if (hsi.zoomme) {
                hsi.icon.style.width = Math.round(hsi.width*zoomXFactor)+"px";
                hsi.icon.style.height = Math.round(hsi.height*zoomYFactor)+"px";
                hsi.w2 = Math.round(hsi.width*zoomXFactor/2);
                hsi.h2 = Math.round(hsi.height*zoomYFactor/2);
              }
              xp = Math.round(((hsi.x0 - xMove )*zoomXFactor - hsi.w2)/canXScale);
              yp = Math.round(((hsi.y0 - yMove )*zoomYFactor - hsi.h2)/canYScale);

              if (hsi.anim) {
                if (xp+hsi.icon.width <= 0 || xp >= imgWidth || yp+hsi.icon.height <= 0 || yp >= imgHeight) continue;

                hsi.icon.style.top = yp+"px";
                hsi.icon.style.left = xp+"px";

              } else {
                ctx.drawImage(hsi.icon,
                    hsi.origX,hsi.origY, hsi.width, hsi.height, xp, yp,
                    hsi.width/canXScale,
                    hsi.height/canYScale);
              }
            } else {
              ctx.drawImage(hsi.icon,
                 hsi.origX,hsi.origY, hsi.width, hsi.height,
                 hsi.x0/canXScale, hsi.y0/canYScale,
                 hsi.width/canXScale, hsi.height/canYScale);
            }
            hsi.icon.style.visibility = "visible";
          }
        }

      }

    } catch (errx) {
      info("Error:"+errx);
    }
  }

  function drawProgress(lab, tot, cnt) {
    if (!showProgress || cnt >= tot) return;
    ctx.save();
    ctx.fillStyle = "blue";
    ctx.strokeStyle = "white";
    ctx.lineWidth = 3;
    ctx.strokeRect(progX,progY,200,18);
    ctx.fillRect(progX,progY,200*(cnt / tot),18);
    var w = ctx.measureText(lab).width;
    ctx.fillStyle = "orange";
    ctx.font = "14px arial";
    ctx.fillText(lab, progX + 100 - w/2, progY+14);
    ctx.restore();
  }

  function drawImageProgress() {
    if (showProgress && imgCount > imgGotCount) {
      drawProgress("Loading Images", imgCount, imgGotCount);

    } else {
      useProgress = false;
      if (loadMsgAuto) {
        loadMsg.style.visibility = "hidden";
      }
    }
    if (imgGotCount >= imgCount && !isLooping) setCurrentFrame(findFrame(curFrame));
    if (cdLoading != undefined && imgGotCount >= imgCount) cdLoading(true);
  }

  function info(s) {
    if (debug && debugWindow && !debugWindow.closed) {
      try {
      debugWindow.document.write(s+"<br>");
      } catch (err) {
      }
    }
  }


  function run() {
      if (isLooping) incCurrentFrame(direction);
      if (curFrame == findFrame(numFrames-1)) {
        delay = lastDwell+dwell;
      } else {
        delay = dwell;
      }
      drawIt();
      setTimeout( function() {
        requestAnimationFrame(run);
      }, delay);

  }

  // The following is for the map projections
  function normalize(lon) {
    while (lon < -180) lon += 360;
    while (lon >  180) lon -= 360;
    return(lon);
  }


  /** @constructor */
  function MORhanCylEqualDist() {
    // code derived from NSIDC's mapx library

    var Rg, lon0, u, v, u0, v0, lat, lon, dlon;
    var RAD = Math.PI/180.0;

    // CE, refLon, origLat, origLon, EquatRad, Spacing, xoff, yoff

    this.init = function(st) {
      var a = [];
      for (var i=1; i<st.length; i++) {
        a[i] = parseFloat(st[i]);
      }

      lon0 = a[1];  // refLon;
      Rg = a[4] / a[5];  // EquatRad / Spacing;
      u0 = a[6];  //xoff;
      v0 = a[7];  //yoff;
      var z = this.toXY(a[2], a[3]);
      u0 = z[0];
      v0 = z[1];
    }

    this.toXY = function(lat, lon) {

      dlon = lon - lon0;
      if (lon*lon0 < 0) {  // cross 180
        dlon = lon < 0 ? lon+360.-lon0 : lon-lon0-360.;
      }
      u = Rg*normalize(lon-lon0)*RAD - u0;
      v = -Rg*lat*RAD - v0;
      return [u,v];

    }

    this.toLatLon = function(u, v) {
      lat = -(v+v0)/Rg/RAD;
      lon = normalize((u+u0)/Rg/RAD + lon0);
      return [lat,lon];
    }

  }

  /** @constructor */
  function MORhanPolarStereoEllips() {
    // code derived from NSIDC's mapx library

    var lat1, lon0, x, y, phi, lam, Rg, cos_phi1, sin_phi1,sin_phi0,
    u,v, lat, lon, u0, v0, t1, m1, t, fact, isNorth, e, e2,
    e4, e6, e8,rho, chi, sin2chi, sin4chi, sin6chi,scale;

    var RAD = Math.PI/180.0;

    // PS, refLat, refLon, origLat, origLon, EquatRad, eccen, Spacing, np, xoff,  yoff

    this.init = function(st) {
      var a = [];
      for (var i=1; i<st.length; i++) {
        a[i] = parseFloat(st[i]);
      }
      lat1 = a[1];
      lon0 = a[2];
      isNorth = (a[8] == 0) ? false : true;

      e = a[6];
      e2 = e*e;
      e4 = e2*e2;
      e6 = e4*e2;
      e8 = e4*e4;

      Rg = a[5] / a[7];
      scale = a[7];

      if (isNorth) {
        cos_phi1 = Math.cos(RAD*lat1);
        sin_phi0 = 1.0; //sin_phi0 = Math.sin(radians(90));
        sin_phi1 = Math.sin(RAD*lat1);
      } else {
        cos_phi1 = Math.cos((-lat1)*RAD);
        sin_phi0 = 1.0; //sin_phi0 = Math.sin(radians(90));
        sin_phi1 = Math.sin((-lat1)*RAD);
      }
      m1 = ((cos_phi1) / Math.sqrt(1.0 - (e2 * sin_phi1 * sin_phi1)));

      fact = (1. - e * sin_phi1) / (1. + e * sin_phi1);

      if(isNorth) {
        t1 = Math.tan(Math.PI / 4. - lat1*RAD / 2.)
            / Math.pow(fact, e / 2.);
      } else {
        t1 = Math.tan(Math.PI / 4. - (-lat1)*RAD / 2.)
            / Math.pow(fact, e / 2.);
      }

      u0 = a[9];
      v0 = a[10];
      var z = this.toXY(a[3], a[4]);
      u0 = z[0];
      v0 = z[1];
    }

    this.toXY = function(lat, lon) {

      if (isNorth) {
        phi = RAD*lat;
        lam = RAD*(lon - lon0);
      } else {
        phi = RAD*(-lat);
        lam = RAD*(lon0 - lon);
      }

      fact = (1. - e * Math.sin(phi)) / (1. + e * Math.sin(phi));
      t = Math.tan(Math.PI / 4. - phi / 2.) / Math.pow(fact, e / 2.);

      if((90.0 != lat1) && (-90.0 != lat1)) {
        rho = Rg * m1 * t / t1;
      } else {
        rho = (2. * Rg * scale * t) /
           (Math.sqrt(Math.pow(1. + e, 1. + e)
               * Math.pow(1. - e, 1. - e)));
      }

      x = rho * Math.sin(lam);
      y = rho * Math.cos(lam);

      if(!isNorth) {
        x = -x;
        y = -y;
      }

      u = x - u0;
      v = y - v0;
      return [u,v]
    }

    this.toLatLon = function(u, v) {

      x = u+u0;
      y = v+v0;

      rho = Math.sqrt(x*x + y*y);

      if(90.0 == lat1 || -90.0 == lat1) {
        t = (rho * Math.sqrt(Math.pow(1 + e, 1 + e)
           * Math.pow(1 - e, 1 - e) )) / (2 * Rg * scale);
      } else {
        t = (rho * t1) / (Rg * m1);
      }
      chi = Math.PI / 2.0 - 2.0 * Math.atan(t);
      sin2chi = Math.sin(2.0 * chi);
      sin4chi = Math.sin(4.0 * chi);
      sin6chi = Math.sin(6.0 * chi);

      phi = chi + (sin2chi * e2 / 2.0) + (sin2chi * 5.0 * e4 / 24.0)
        + (sin2chi * e6 / 12.0) + (sin2chi * 13.0 * e8 / 360.0)
        + (sin4chi * 7.0 * e4 / 48.0) + (sin4chi * 29.0 * e6 / 240.0)
        + (sin4chi * 811.0 * e8 / 11520.0)
        + (sin6chi * 7.0 * e6 / 120.0)
        + (sin6chi * 81.0 * e8 / 1120.0)
        + (Math.sin(8.0 * chi) * 4279.0 * e8 / 161280.0);

      if(isNorth) {
        lat = phi/RAD;
        lon = Math.atan2(x, y)/RAD + lon0;

      } else {
        lat = -phi/RAD;
        lon = -Math.atan2(-x, -y)/RAD + lon0;
      }

      return [lat, normalize(lon)];
    }
  }

  /** @constructor */
  function MORhanLambConConEllips() {
    // code derived from NSIDC mapx library

    var lat0, lat1, lon0, x, y, phi, lam, Rg, cos_phi0,cos_phi1, sin_phi1,sin_phi0;
    var u,v,lat,lon,u0, v0, t0, t1, m0, m1, t, e, e2, e4, e6, e8;
    var chi, rho, n, F, theta, rho0, sin_phi;
    var RAD = Math.PI/180.0;

    //LCC, refLat0, refLat1, refLon, origLat, origLon, EquatRad, eccen, Spacing, xoff,  yoff

    this.init = function(st) {

      var a = [];
      for (var i=1; i<st.length; i++) {
        a[i] = parseFloat(st[i]);
      }

      lat0 = a[1];
      lat1 = a[2];
      lon0 = a[3];
      e = a[7];
      e2 = e*e;
      e4 = e2*e2;
      e6 = e4*e2;
      e8 = e4*e4;

      Rg = a[6] / a[8];
      cos_phi0 = Math.cos(RAD*lat0);
      cos_phi1 = Math.cos(RAD*lat1);
      sin_phi0 = Math.sin((lat0)*RAD);
      sin_phi1 = Math.sin((lat1)*RAD);

      m0 = ((cos_phi0)/Math.sqrt(1 - (e2 * sin_phi0 * sin_phi0)));
      m1 = ((cos_phi1)/Math.sqrt(1 - (e2 * sin_phi1 * sin_phi1)));
      t0 = Math.sqrt( ((1.0 - sin_phi0)/(1.0 + sin_phi0)) *
        Math.pow(((1.0 + (e* sin_phi0))/(1.0 - (e* sin_phi0))),e) );
      t1 = Math.sqrt( ((1.0 - sin_phi1)/(1.0 + sin_phi1)) *
        Math.pow(((1.0 +(e* sin_phi1))/ (1.0 - (e* sin_phi1))), e) );
      n = (Math.log(m0) - Math.log(m1)) / (Math.log(t0) - Math.log(t1));
      F = m0/(n * Math.pow(t0,n));
      rho0 = Rg * F * Math.pow(t0, n);

      u0 = a[9];
      v0 = a[10];
      var z = this.toXY(a[4], a[5]);
      u0 = z[0];
      v0 = z[1];
    }

    this.toXY = function(lat, lon) {

      phi = RAD*lat;
      lam = RAD*normalize(lon - lon0);

      sin_phi = Math.sin(phi);

      t = Math.sqrt( ((1. - sin_phi)/(1. + sin_phi)) *
             Math.pow(((1. + (e* sin_phi))/ (1. - (e * sin_phi))), e) );

      rho = Rg * F * Math.pow(t, n);
      theta = n * lam;

      x = rho * Math.sin(theta);
      y = rho0 + (rho * Math.cos(theta));

      u = x - u0;
      v = y - v0;
      return [u,v]
    }

    this.toLatLon = function(u, v) {

      x = u+u0;
      y = v+v0;
      rho = (Math.abs(n) / n) * Math.sqrt((x*x) + ((rho0 - y) * (rho0 - y)));
      t = Math.pow((rho/(Rg * F)), (1/n));
      chi = Math.PI/2.0 - 2.0 * Math.atan(t);

      if (n < 0.0) {
        theta = Math.atan( -x / (y - rho0));
      } else {
        theta = Math.atan(x / (rho0 - y));
      }

      lam = (theta / n);
      phi = chi + (((e2 / 2.0) + ((5.0 / 24.0) * e4) +
            (e6 / 12.0) + ((13.0 / 360.0) * e8)) * Math.sin(2.0 * chi)) +
        ((((7.0 / 48.0) * e4) + ((29.0 / 240.0) * e6) +
          ((811.0 / 11520.0) * e8)) * Math.sin(6.0 * chi)) +
        ((((7.0 / 120.0) * e6) +
          ((81.0 / 1120.0) * e8)) * Math.sin(6.0 * chi)) +
        (((4279.0 / 161280.0) * e8) * Math.sin(8.0 * chi));

      lat = phi/RAD;
      lon = normalize(lon0 - lam/RAD );
      return [lat, lon];
    }
  }

}


