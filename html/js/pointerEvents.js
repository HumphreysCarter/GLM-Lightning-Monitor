"use strict";
/** @constructor */
function PEvs (ele, down, up, move, drag, click, stopped) {
  var touches, first, isTouching, isDown, downTime, upTime, durTime;
  var element, offsetLeft, offsetTop, didMove, sentStop, timer;
  var callUp, callDown, callDrag, callMove, callClick, callStopped;
  var callWheel;
  var x = 0;
  var y = 0;
  var oldX, oldY;
  var ptrID, isMulti = false, isPinch = false;;
  var dist=0, oldDist=0, dx=0,dy=0, avgX, avgY;

  var myele = ele;

  var getPosition = function(event) {
    element = event.target; 
    offsetLeft=0; 
    offsetTop=0;
    while (element) {
      offsetLeft += (element.offsetLeft - element.scrollLeft);
      offsetTop += (element.offsetTop - element.scrollTop);
      element = element.offsetParent;
    }
    
    x = (isMulti ? avgX : event.pageX) - offsetLeft - document.body.scrollLeft;
    y = (isMulti ? avgY : event.pageY) - offsetTop - document.body.scrollTop;  

  };

  this.getX = function() { 
    return x; 
  }
  this.getY = function() { 
    return y; 
  }

  var buck = function(e) {
    e.preventDefault();
    if (e.stopImmediatePropagation) {
      e.stopImmediatePropagation();
    } else {
      e.stopPropagation();
    }
  }

  var doDown = function(e) {
    getPosition(e);
    oldX = x;
    oldY = y;
    isDown = true;
    if (callClick != null) downTime  = new Date().getTime();
    if (callDown != null) callDown(e);  // get er rolling
  }

  var doUp = function(e) {
    if (!isDown) return;
    getPosition(e);
    isDown = false;
    if (callUp != null) callUp(e);
    if (callClick != null) {
      upTime = (new Date()).getTime();
      durTime = upTime - downTime;
      if (durTime < 300) callClick(e);
    }
  }

  var doMove = function(e) {
    getPosition(e);
    if (Math.abs(x - oldX) < 3 && Math.abs(y - oldY) < 3) {
      return;
    }

    oldX = -9999;
    oldY = -9999;

    if (isDown) {
      if (callDrag != null) callDrag(e);
    } else {
      if (callMove != null) callMove(e);
    }
    if (callStopped != null) doDidMove();
  }

  this.touchDown = function(e) {
    isTouching = true;
    if (e.targetTouches.length > 1) {
      isMulti = true;
      if (callWheel) {
        oldDist = -1;
        buck(e);
      }
      return false;
    }

    touches = e.changedTouches; 
    first = touches[0];
    doDown(first);
  };

  this.touchUp = function(e) {
    if (e.targetTouches.length == 0) {
      isTouching = false;
      isMulti = false;
    }
    buck(e);
    if (isMulti || isPinch) {
      if (!isMulti) isPinch = false;
      return false;
    }
    touches = e.changedTouches; 
    first = touches[0];
    doUp(first);
  };

  this.touchMove = function(e) {
    isTouching = true;
    buck(e);

    if (isMulti) {
      if (callWheel) {
        touches = e.targetTouches;
        avgX = (touches[0].pageX + touches[1].pageX)/2;
        avgY = (touches[0].pageY + touches[1].pageY)/2;
        isPinch = true;
        dx = touches[0].pageX - touches[1].pageX;
        dy = touches[0].pageY - touches[1].pageY;
        dist = dx*dx + dy*dy;
        if (callWheel && (oldDist > 0) && (Math.abs(dist-oldDist) > 30)) {
          callWheel( (dist - oldDist)>0?1:-1);
        }
        oldDist = dist;
      }
      return false;
    }

    touches = e.changedTouches;
    first = touches[0];
    doMove(first);
  };

  // begin pointerEvents
  this.pointUp = function(e) {
    buck(e);
    for (var i=0; i<ptrID.length; i++) {
      if (ptrID[i].pointerId == e.pointerId) {
        ptrID.splice(i,1);
        break;
      }
    }
    //isMulti = false;
    if (ptrID.length == 0) {
      isMulti = false;
      isTouching = false;
    }
    if (isMulti || isPinch) {
      if (!isMulti) isPinch = false;
      return false;
    }
    doUp(e);
    return false;
  }

  this.pointDown = function(e) {
    buck(e);
    ptrID.push(e);
    oldDist = -1;
    isTouching = true;
    if (ptrID.length > 1) {
      isMulti = true;
    } else {
      doDown(e);
    }
    return false;
  }

  this.pointMove = function(e) {
    buck(e);
    // update target
    for (var i=0; i<ptrID.length; i++) {
      if (ptrID[i].pointerId == e.pointerId) ptrID[i] = e;
    }
    if (!isMulti && isPinch) return;
    if (isMulti) {
      avgX = (ptrID[0].pageX + ptrID[1].pageX)/2;
      avgY = (ptrID[0].pageY + ptrID[1].pageY)/2;
      if (callWheel) {
        isPinch = true;
        dx = ptrID[0].pageX - ptrID[1].pageX;
        dy = ptrID[0].pageY - ptrID[1].pageY;
        dist = dx*dx + dy*dy;
        if (callWheel && (oldDist > 0) && (Math.abs(dist-oldDist) > 30)) {
          callWheel( (dist - oldDist)>0?1:-1);
        }
        oldDist = dist;
      }
      return false;
    }

    doMove(e);
    return false;
  }
  // end pointerEvents


  this.mouseDown = function(e) {
    if (isTouching) return;
    buck(e);
    doDown(e);
  };

  this.clearDown = function() {
    isDown = false;
  }

  this.mouseUp = function(e) {
    if (isTouching) return;
    buck(e);
    doUp(e);
  };

  this.useWheel = function(cw) {
    myele.addEventListener("wheel", this.mouseWheel, true);
    callWheel = cw;
  }

  this.mouseWheel = function(e) {
    getPosition(e);
    var delta = Math.max(-1, Math.min(1, (-e.deltaY || -e.detail)));
    buck(e);
    if (callWheel) callWheel(delta);
  };

  this.mouseClick = function(e) {
    buck(e);
    isDown = false;
  };

  this.mouseMove = function(e) {
    if (isTouching) return;
    buck(e);
    doMove(e);
  };

  this.mouseDragged = function(e) {
    if (isTouching) return;
    buck(e);
    getPosition(e);
    if (callDrag != null) callDrag(e);
    if (callStopped != null) doDidMove();
  };

  var doDidMove = function() {
    clearTimeout(timer);
    timer = setTimeout( function() {
      callStopped();
    },600 );
  }

  x = 0;
  y = 0;
  oldX = 0;
  oldY = 0;
  callDown = down;
  callUp =up;
  callMove = move;
  callDrag = drag;
  callClick = click;
  callStopped = stopped;
  isTouching = false;
  isDown = false;
  didMove = false;
  sentStop = true;
  ptrID = [];
  ele.addEventListener("mousedown",this.mouseDown,false);
  ele.addEventListener("mouseup", this.mouseUp, false);
  ele.addEventListener("mousemove",this.mouseMove, false);
  ele.addEventListener("click",this.mouseClick, false);

  // Ongoing issue with Firefox with Pointer Events
  //if (window.PointerEvent) {
  //  ele.addEventListener("pointerdown",this.pointDown, false);
  //  ele.addEventListener("pointerup", this.pointUp, false );
  //  ele.addEventListener("pointermove",this.pointMove, false);
  //  ele.addEventListener("pointerleave", this.pointUp, false );
  //  ele.addEventListener("pointercancel", this.pointUp, false );
  //} else {
 
    ele.addEventListener("touchstart",this.touchDown,false);
    ele.addEventListener("touchend",this.touchUp,false);
    ele.addEventListener("touchmove",this.touchMove,false);
    ele.addEventListener("touchleave", this.touchUp, false);
    ele.addEventListener("touchcancel", this.touchUp, false);

  // }
  
  ele.addEventListener("contextmenu", function(e) {
    e.preventDefault();
  }, false);


  ele.style["touch-action"] = "none";

  ele.addEventListener("mouseout", function(e) {
    if (!isDown) {
      x = -1;
      y = -1;
      if (callMove != null) callMove(e); 
    }
    isDown = false;
  });

}
