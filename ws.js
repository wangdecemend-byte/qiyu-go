// ============================================================
// ws.js — WebSocket 联网对战模块（独立封装，零污染）
// 棋语 v2.0
// ============================================================
(function () {
  'use strict';

  // ── 配置 ──
  // WS 地址：优先 URL 参数 ?ws=xxx，其次自动检测
  var DEFAULT_WS = (function () {
    var u = new URL(window.location.href);
    var param = u.searchParams.get('ws');
    if (param) return param;
    // file:// 或 localhost 本地开发 → 直连 localhost:8080
    if (u.protocol === 'file:' || u.hostname === 'localhost' || u.hostname === '127.0.0.1') {
      return 'ws://localhost:8080';
    }
    // 生产环境：前端静态站 → 连后端 WS 服务器
    if (u.hostname === 'qiyu-go.onrender.com') {
      return 'wss://half-day-chess-server.onrender.com';
    }
    // 其他线上部署 → 同主机
    var protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
    return protocol + '//' + u.host;
  })();

  var MAX_RECONNECT_DELAY = 30000;   // 最大重连间隔 30s
  var INITIAL_RECONNECT_DELAY = 1000; // 初始重连间隔 1s

  // ── 内部状态 ──
  var ws = null;
  var reconnectTimer = null;
  var reconnectDelay = INITIAL_RECONNECT_DELAY;
  var status = 'disconnected'; // 'connecting' | 'connected' | 'disconnected'
  var messageHandlers = {};    // { type: [handler, ...] }
  var globalHandlers = [];     // 所有消息都会触发
  var statusHandlers = [];     // 状态变化回调
  var pendingQueue = [];       // 断线期间暂存的消息队列（最多 50 条）
  var intentionalClose = false;// 是否主动断开

  // ── 工具函数 ──
  function _notifyStatus(newStatus) {
    if (status === newStatus) return;
    var prev = status;
    status = newStatus;
    for (var i = 0; i < statusHandlers.length; i++) {
      try { statusHandlers[i](status, prev); } catch (e) {}
    }
  }

  function _emit(type, data) {
    // 触发特定类型处理器
    var handlers = messageHandlers[type];
    if (handlers) {
      for (var i = 0; i < handlers.length; i++) {
        try { handlers[i](data); } catch (e) {}
      }
    }
    // 触发全局处理器
    for (var j = 0; j < globalHandlers.length; j++) {
      try { globalHandlers[j](type, data); } catch (e) {}
    }
  }

  function _flushQueue() {
    if (pendingQueue.length === 0) return;
    while (pendingQueue.length > 0) {
      var msg = pendingQueue.shift();
      _rawSend(msg);
    }
  }

  function _flushNonGame() {
    // 只重放非游戏消息（聊天类），游戏操作不重放避免状态冲突
    var remaining = [];
    while (pendingQueue.length > 0) {
      var msg = pendingQueue.shift();
      if (msg.type === 'move' || msg.type === 'pass' || msg.type === 'resign') {
        remaining.push(msg);
      } else {
        _rawSend(msg);
      }
    }
    // 将未发送的游戏消息放回队列（如果后续确认需要可手动重发）
    pendingQueue = remaining;
  }

  function _rawSend(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(typeof msg === 'string' ? msg : JSON.stringify(msg));
      return true;
    }
    return false;
  }

  // ── 连接管理 ──
  function connect(url) {
    var target = url || DEFAULT_WS;

    // 如果已经连接或正在连接，先断开
    if (ws) {
      intentionalClose = true;
      try { ws.close(1000, 'reconnect'); } catch (e) {}
      ws = null;
    }

    intentionalClose = false;
    _notifyStatus('connecting');

    try {
      ws = new WebSocket(target);
    } catch (e) {
      _notifyStatus('disconnected');
      _scheduleReconnect();
      return;
    }

    ws.onopen = function () {
      reconnectDelay = INITIAL_RECONNECT_DELAY;
      _notifyStatus('connected');
      // 只重放非游戏类消息（聊天等），游戏消息让应用层决定是否重同步
      _flushNonGame();
      // 通知应用层连接已恢复，需同步棋局
      _emit('reconnected', {});
    };

    ws.onmessage = function (e) {
      var data;
      try { data = JSON.parse(e.data); } catch (err) {
        // 非 JSON 消息，忽略
        return;
      }
      if (data && data.type) {
        _emit(data.type, data);
      }
    };

    ws.onclose = function (e) {
      ws = null;
      _notifyStatus('disconnected');
      if (!intentionalClose) {
        _scheduleReconnect();
      }
      _emit('close', { code: e.code, reason: e.reason });
    };

    ws.onerror = function () {
      // onclose 会紧随其后触发，由 onclose 统一处理重连
    };
  }

  // 初始自动连接（仅在非 file:// 协议时，因为本地打开无法连接远程 WS）
  if (window.location.protocol !== 'file:') {
    connect(DEFAULT_WS);
  }

  function disconnect() {
    intentionalClose = true;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (ws) {
      try { ws.close(1000, 'user_disconnect'); } catch (e) {}
      ws = null;
    }
    _notifyStatus('disconnected');
    pendingQueue = [];
  }

  function _scheduleReconnect() {
    if (intentionalClose) return;
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      connect();
      // 指数退避：1s → 2s → 4s → 8s → 16s → 30s (max)
      reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
    }, reconnectDelay);
  }

  // ── 发送消息 ──
  function send(msg) {
    if (!msg || !msg.type) return false;

    // 断线时入队（匹配类消息不入队，立即返回失败）
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      // 游戏中的消息才入队
      if (msg.type === 'move' || msg.type === 'pass' || msg.type === 'resign') {
        if (pendingQueue.length < 50) {
          pendingQueue.push(msg);
        }
      }
      return false;
    }

    return _rawSend(msg);
  }

  // ── 事件订阅 ──
  function on(type, handler) {
    if (!messageHandlers[type]) messageHandlers[type] = [];
    messageHandlers[type].push(handler);
  }

  function off(type, handler) {
    var handlers = messageHandlers[type];
    if (!handlers) return;
    var idx = handlers.indexOf(handler);
    if (idx >= 0) handlers.splice(idx, 1);
  }

  function onStatus(handler) {
    statusHandlers.push(handler);
  }

  function onAny(handler) {
    globalHandlers.push(handler);
  }

  // ── 便捷方法 ──
  function isConnected() {
    return status === 'connected';
  }

  function getStatus() {
    return status;
  }

  function getReconnectDelay() {
    return reconnectDelay;
  }

  // ============ 暴露全局 API ============
  window.WS = {
    // 连接
    connect: connect,
    disconnect: disconnect,
    isConnected: isConnected,
    getStatus: getStatus,

    // 消息
    send: send,
    on: on,
    off: off,
    onStatus: onStatus,
    onAny: onAny,

    // 便捷游戏方法
    match: function () {
      return send({ type: 'match' });
    },
    createRoom: function () {
      return send({ type: 'create' });
    },
    joinRoom: function (code) {
      return send({ type: 'join', room: String(code).toUpperCase() });
    },
    sendMove: function (x, y, color) {
      return send({ type: 'move', x: x, y: y, color: color });
    },
    sendPass: function () {
      return send({ type: 'pass' });
    },
    sendResign: function () {
      return send({ type: 'resign' });
    },

    // 调试
    _resetReconnectDelay: function () {
      reconnectDelay = INITIAL_RECONNECT_DELAY;
    },
  };

})();
