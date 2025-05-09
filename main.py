import asyncio
import struct
import websockets
from typing import Callable, Any, Optional
import header_pb2
import lobby_pb2
import dzpk_pb2
from enum import Enum
from loguru import logger
# import solver

logger.add("poker.log", rotation="100 MB")  # 自动轮转日志

def convert_card(card_in):
    color = "d"
    num = 0
    if card_in <= 13:
        # 方片
        # color = "diamond"
        color = "d"
        num = card_in
    elif card_in <= 29:
        # 梅花
        # color = "club"
        color = "c"
        num = card_in - 16
    elif card_in <= 45:
        # 红桃
        # color = "heart"
        color = "h"
        num = card_in - 32
    elif card_in <= 61: 
        # 黑桃
        # color = "spade"
        color = "s"
        num = card_in - 48

    if num == 1:
        num = 14
        return 'A' + color
    elif num == 10:
        return 'T' + color    
    elif num == 11:
        return 'J' + color
    elif num == 12:
        return 'Q' + color
    elif num == 13:
        return 'K' + color
    else:
        return str(num) + color


class RoomStatus(Enum):
    RoomStatusNone = 0
    RoomStatusWaitPlayers = 1                  # 等待足够玩家进入
    RoomStatusWaitPrepare = 2                  # 等待各位玩家准备（当房间内人数已经够了时，就会让各位玩家准备，同时开始倒计时）
    RoomStatusStarted = 3                      # 刚开局
    RoomStatusPlayCards = 4                    # 已开始打牌
    RoomStatusPreFlop = 5                      # 翻前状态
    RoomStatusFlop = 6                         # 翻牌状态
    RoomStatusTurn = 7                         # 转牌状态
    RoomStatusRiver = 8                        # 河牌状态
    RoomStatusShowDown = 9                     # 摊牌状态
    RoomStatusGameOver = 10                    # 一局结束

class Op(Enum):
    Belt     = 11   # 下注
    Call     = 12   # 跟注
    Raise     = 13   # 加注
    DisCard  = 14   # 弃牌
    Check     = 15   # 过牌
    AllIn    = 16   # 下全注

class WebSocketClient:
    def __init__(self):
        self.socket = None
        self.uri = 'ws://81.69.249.211:19001'
        self._callbacks = {
            'on_login': self.on_login,
            'on_message': self.on_message,
            'on_get_my_data' : self.on_get_my_data,
            'on_dispatch_start_cards' : self.on_dispatch_start_cards,
            'on_enter_room' : self.on_enter_room,
            'on_room_snapshot' : self.on_room_snapshot,
            'on_game_start' : self.on_game_start,
            'on_blind' : self.on_blind,
            'on_op_list' : self.on_op_list,
            'on_stop_timer' : None,
            'on_start_timer' : None,
            'on_call' : self.on_call,
            'on_look_card' : self.on_look_card,
            'on_start_flop' : self.on_start_flop,
            'on_start_turn' : self.on_start_turn,
            'on_start_river' : self.on_start_river,
            'on_compare_cards' : None,
            'on_diss_card' : self.on_diss_card,
            'on_belt' : self.on_belt,
            'on_fill' : self.on_fill,
            'on_all_in' : self.on_all_in,
            'on_lose_and_win_info' : None,
            'on_arrange_players_chip' : self.on_arrange_players_chip,
            'on_start_pre_flop' : None,
            'on_hold_desk' : None,
            'on_leave_room' : None,
            'on_add_chips_suc' : self.on_add_chips_suc,
            'on_enter_my_room' : self.on_enter_my_room,
        }
        self._running = False
        self.my_uid = 0    # 我的玩家id
        self.my_hand_cards = []    # 手牌
        self.room_status = RoomStatus.RoomStatusWaitPlayers    # 房间状态
        self.table_cards = []    # 公共牌
        self.my_seat_number = 0  # 我的座位号
        self.bank_id = 0         # 庄家id
        self.small_blind = 0     # 小盲
        self.big_blind = 0       # 大盲
        self.cur_step_belt_time = 0    # 本轮第几次下注
        self.available_ops = None
        self.available_chip = 0
        self.money_on_the_table = 0
        self.g_pot = 0
        self.g_player_num = 0
        self.action_done = []    # 需要记录上个玩家的操作
        self.dict_money_on_table = {}
        self.room_no = ""
        self.room_password = ""

    def register_callback(self, event: str, callback: Callable[[Any], None]):
        """Register a callback for specific events
        
        Available events:
        - on_login: Triggered when login response received
        - on_message: Triggered when any message received
        - on_error: Triggered when error occurs
        - on_disconnect: Triggered when connection closed
        """
        if event in self._callbacks:
            self._callbacks[event] = callback
        else:
            raise ValueError(f"Unknown event type: {event}")

    async def connect(self, uri: str = None):
        """Connect to WebSocket server"""
        if uri:
            self.uri = uri
        
        try:
            self.socket = await websockets.connect(self.uri)
            self._running = True
            asyncio.create_task(self._receive_loop())
            logger.info("Connected to {}", self.uri)
        except Exception as e:
            self._trigger_callback('on_error', e)
            raise

    async def disconnect(self):
        """Disconnect from server"""
        self._running = False
        # if self.socket and not self.socket.closed:
        if self.socket:
            await self.socket.close()
            self._trigger_callback('on_disconnect')

    #设置要进入的房间号和密码
    def set_room_no_pwd(self, room_no, room_pwd):
        self.room_no = room_no
        self.room_password = room_pwd

    async def send_message(self, header_stream: bytes, body_stream: bytes) -> bool:
        """Send a protobuf message to server"""
        # if not self.socket or self.socket.closed:
        if not self.socket:
            self._trigger_callback('on_error', "Connection not established")
            return False

        try:
            header_length = len(header_stream)
            body_length = len(body_stream)
            out_stream = struct.pack('>II', header_length, body_length) + header_stream + body_stream
            await self.socket.send(out_stream)
            return True
        except Exception as e:
            self._trigger_callback('on_error', e)
            return False

    async def send_login_msg(self, name: str, password: str) -> bool:
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csproto.CSAuth"
        header.gameId = "dzpk"

        # Prepare body
        body = lobby_pb2.CSAuth()
        body.name = name
        body.password = password
        # Set other required fields
        body.token = "123"
        body.clientVersion = "1.0.0"
        body.isMobile = False
        # ... set other fields as needed

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)
    
    async def send_get_my_data(self) -> bool:
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csproto.CSGetMyData"
        header.gameId = "dzpk"

        # Prepare body
        body = lobby_pb2.CSGetMyData()

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)
    
    async def send_enter_room(self) -> bool:
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csdzpkproto.CSEnterRoom"
        header.gameId = "dzpk"

        # Prepare body
        body = dzpk_pb2.CSEnterRoom()
        body.roomLevel = 2

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)
    
    async def send_enter_my_room(self) -> bool:
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csproto.CSEnterMyRoom"
        header.gameId = "dzpk"

        # Prepare body
        body = lobby_pb2.CSEnterMyRoom()

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)
    
    async def send_enter_friend_room(self) -> bool:
        # logger.info('send_enter_friend_room----------------------------------------------')
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csdzpkproto.CSEnterFriendRoom"
        header.gameId = "dzpk"

        # Prepare body
        body = dzpk_pb2.CSEnterFriendRoom()
        body.roomNo = self.room_no
        body.password = self.room_password

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)
    
    async def send_room_scene_prepared(self) -> bool:
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csdzpkproto.CSRoomScenePrepared"
        header.gameId = "dzpk"

        # Prepare body
        body = dzpk_pb2.CSRoomScenePrepared()

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)
    
    async def send_cancel_hold_desk(self, uid) -> bool:
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csdzpkproto.CSCancelHoldDesk"
        header.gameId = "dzpk"

        # Prepare body
        body = dzpk_pb2.CSCancelHoldDesk()
        body.uid = uid

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)
    
    # 跟注
    async def send_call(self) -> bool:
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csdzpkproto.CSCall"
        header.gameId = "dzpk"

        # Prepare body
        body = dzpk_pb2.CSCall()
        body.chip = 0

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)
    
    # 加注
    async def send_fill(self, chip) -> bool:
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csdzpkproto.CSFill"
        header.gameId = "dzpk"

        # Prepare body
        body = dzpk_pb2.CSFill()
        body.chip = chip

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)
    
    # 下注
    async def send_belt(self, chip) -> bool:
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csdzpkproto.CSBelt"
        header.gameId = "dzpk"

        # Prepare body
        body = dzpk_pb2.CSBelt()
        body.chip = chip

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)
    
    # all in
    async def send_all_in(self) -> bool:
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csdzpkproto.CSAllIn"
        header.gameId = "dzpk"

        # Prepare body
        body = dzpk_pb2.CSAllIn()

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)
    
    # 过牌
    async def send_pass(self) -> bool:
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csdzpkproto.CSLookCard"
        header.gameId = "dzpk"

        # Prepare body
        body = dzpk_pb2.CSLookCard()

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)


    async def send_give_up(self) -> bool:
        # Prepare header
        header = header_pb2.Header()
        header.messageName = "csdzpkproto.CSDissCard"
        header.gameId = "dzpk"

        # Prepare body
        body = dzpk_pb2.CSDissCard()

        # Serialize and send
        header_stream = header.SerializeToString()
        body_stream = body.SerializeToString()
        return await self.send_message(header_stream, body_stream)

    async def _receive_loop(self):
        """Main receive loop for handling incoming messages"""
        while self._running and self.socket:
            try:
                message = await self.socket.recv()
                
                if len(message) < 8:
                    continue
                
                header_len, body_len = struct.unpack('>II', message[:8])
                if len(message) < 8 + header_len + body_len:
                    continue
                
                header_data = message[8:8+header_len]
                body_data = message[8+header_len:8+header_len+body_len]
                
                header = header_pb2.Header()
                header.ParseFromString(header_data)
                
                # 使用 await 调用
                await self._trigger_callback('on_message', {
                    'header': header,
                    'body_data': body_data
                })

                # 不需要同步示例
                # if self._callbacks['on_message']:
                #     self._callbacks['on_message']({
                #     'header': header,
                #     'body_data': body_data
                # })
                # logger.info('----------------------------------------{}', header.messageName)
                
                if header.messageName == "csproto.SCAuth":
                    body = lobby_pb2.SCAuth()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_login', body)
                elif header.messageName == "csproto.SCGetMyData":
                    body = lobby_pb2.SCGetMyData()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_get_my_data', body)
                elif header.messageName == "csdzpkproto.SCDispatchStartCards":
                    body = dzpk_pb2.SCDispatchStartCards()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_dispatch_start_cards', body)
                elif header.messageName == "csdzpkproto.SCDzOpList":
                    body = dzpk_pb2.SCDzOpList()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_op_list', body)
                elif header.messageName == "csdzpkproto.SCStartFlop":
                    body = dzpk_pb2.SCStartFlop()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_start_flop', body)
                elif header.messageName == "csdzpkproto.SCStartTurn":
                    body = dzpk_pb2.SCStartTurn()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_start_turn', body)
                elif header.messageName == "csdzpkproto.SCStartRiver":
                    body = dzpk_pb2.SCStartRiver()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_start_river', body)
                elif header.messageName == "csdzpkproto.SCGameStart":
                    body = dzpk_pb2.SCGameStart()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_game_start', body)
                elif header.messageName == "csdzpkproto.SCBlind":
                    body = dzpk_pb2.SCBlind()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_blind', body)
                elif header.messageName == "csdzpkproto.SCEnterRoom":
                    body = dzpk_pb2.SCEnterRoom()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_enter_room', body)
                elif header.messageName == "csdzpkproto.SCLookCard":
                    body = dzpk_pb2.SCLookCard()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_look_card', body)
                elif header.messageName == "csdzpkproto.SCCall":
                    body = dzpk_pb2.SCCall()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_call', body)
                elif header.messageName == "csdzpkproto.SCBelt":
                    body = dzpk_pb2.SCBelt()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_belt', body)  
                elif header.messageName == "csdzpkproto.SCFill":
                    body = dzpk_pb2.SCFill()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_fill', body)
                elif header.messageName == "csdzpkproto.SCAllIn":
                    body = dzpk_pb2.SCAllIn()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_all_in', body)
                elif header.messageName == "csdzpkproto.SCAddChipsSuc":
                    body = dzpk_pb2.SCAddChipsSuc()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_add_chips_suc', body)
                elif header.messageName == "csdzpkproto.SCArrangePlayersChip":
                    body = dzpk_pb2.SCArrangePlayersChip()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_arrange_players_chip', body)
                elif header.messageName == "csdzpkproto.SCDissCard":
                    body = dzpk_pb2.SCDissCard()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_diss_card', body)
                elif header.messageName == "csdzpkproto.SCRoomSnapshot":
                    body = dzpk_pb2.SCRoomSnapshot()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_room_snapshot', body)
                elif header.messageName == "csproto.SCEnterMyRoom":
                    body = lobby_pb2.SCEnterMyRoom()
                    body.ParseFromString(body_data)
                    await self._trigger_callback('on_enter_my_room', body)



            except websockets.exceptions.ConnectionClosed:
                self._running = False
                await self._trigger_callback('on_disconnect')
            except Exception as e:
                self._running = False
                await self._trigger_callback('on_error', e)

    async def _trigger_callback(self, event: str, *args):
        """Internal method to trigger callbacks"""
        if event in self._callbacks and self._callbacks[event]:
            try:
                callback = self._callbacks[event]
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                logger.info("Error in {} callback:", e)

    def get_most_money_on_table(self):
        most_money = 0
        for key in self.dict_money_on_table.keys():
            if self.dict_money_on_table[key] > most_money:
                most_money = self.dict_money_on_table[key]
        return most_money
    
    async def on_login(self, data):
        # logger.info("on_login: ", data)
        self.my_uid = data.uid
        await self.send_get_my_data()

    def on_message(self, data):
        # logger.info("\non_message------------------------------------:", data['header'].messageName)
        # logger.info(f"------------------------------------Message Type: {data['header'].messageName}")
        pass

    async def on_get_my_data(self, data):
        await self.send_enter_friend_room()
        pass

    async def on_enter_room(self, data):
        logger.info('on_enter_room:{}', data)
        if data.roomPlayer.uid == self.my_uid:
            self.my_seat_number = data.roomPlayer.seatNumber
            self.available_chip = data.roomPlayer.takeChip

            if data.roomPlayer.isHoldDesk or data.roomPlayer.isHoldDesked:
                await self.send_cancel_hold_desk(self.my_uid)

        await self.send_room_scene_prepared()

    async def on_game_start(self, data):
        # logger.info('on_game_start:', data)
        self.bank_id = data.bankerId
        self.small_blind = data.SmallBlind
        self.big_blind = data.BigBlind
        self.g_pot = 0
        self.money_on_the_table = 0
        for key in self.dict_money_on_table.keys():
            self.dict_money_on_table[key] = 0

    async def on_blind(self, data):
        # logger.info('on_game_start:', data)
        if data.uid == self.my_uid:
            self.available_chip -= data.chip
            self.money_on_the_table += data.chip
        self.g_pot += data.chip
        if data.uid in self.dict_money_on_table.keys():
            self.dict_money_on_table[data.uid] += data.chip
        else:
            self.dict_money_on_table[data.uid] = data.chip

    async def on_dispatch_start_cards(self, data):
        # logger.info('on_dispatch_start_cards: ', data)
        self.my_hand_cards = []
        self.table_cards = []
        self.room_status = RoomStatus.RoomStatusPreFlop
        self.action_done = []

        self.g_player_num = 0
        for player_cards in data.playerCards:
            self.g_player_num += 1
            if player_cards.uid == self.my_uid:
                self.my_hand_cards.append(convert_card(player_cards.handCards[0]))
                self.my_hand_cards.append(convert_card(player_cards.handCards[1]))

    async def choice_operation(self, opButtons):
        if len(opButtons) == 0:
            return

        # 获取当前可用的操作按钮
        self.available_ops = [Op(op.buttonId) for op in opButtons]
    
        if Op.Call in self.available_ops:
            await self.send_call()
            return        
        if Op.Check in self.available_ops:
            await self.send_pass()
            return
        if Op.AllIn in self.available_ops:
            await self.send_all_in()
            return
        if Op.DisCard in self.available_ops:
            await self.send_give_up()
            return


    async def on_op_list(self, data):
        # logger.info('small blind:{}', self.small_blind)
        logger.info('uid:{}, available_chip:{}, pot:{}', self.my_uid, self.available_chip, self.g_pot)
        logger.info("Hand cards: {}, Table cards: {}", self.my_hand_cards, self.table_cards)
        
        if not data.opButtons:
            return
        self.cur_step_belt_time += 1
        await self.choice_operation(data.opButtons)

    # 翻牌阶段
    async def on_start_flop(self, data):
        # logger.info('on_start_flop: ', data)       
        self.action_done = []
        self.room_status = RoomStatus.RoomStatusFlop
        self.cur_step_belt_time = 0
        self.table_cards = []
        for card in data.cards:
            self.table_cards.append(convert_card(card))
        for key in self.dict_money_on_table.keys():
            self.dict_money_on_table[key] = 0

    # 转牌
    async def on_start_turn(self, data):
        self.action_done = []
        self.room_status = RoomStatus.RoomStatusTurn
        self.cur_step_belt_time = 0
        self.table_cards.append(convert_card(data.card))
        for key in self.dict_money_on_table.keys():
            self.dict_money_on_table[key] = 0

    # 河牌
    async def on_start_river(self, data):
        self.action_done = []
        self.room_status = RoomStatus.RoomStatusRiver        
        self.cur_step_belt_time = 0
        self.table_cards.append(convert_card(data.card))
        for key in self.dict_money_on_table.keys():
            self.dict_money_on_table[key] = 0

    # 有玩家看牌
    async def on_look_card(self, data):
        operation = {'uid': data.uid, 'op':Op.Check}
        self.action_done.append(operation)
        logger.info('on_look_card:{}', operation)

    # 有玩家call
    async def on_call(self, data):
        operation = {'uid': data.uid, 'op':Op.Call, 'chip': data.chip}
        self.action_done.append(operation)
        logger.info('on_call:{}', operation)

        if data.uid == self.my_uid:
            self.available_chip -= data.chip
            self.money_on_the_table += data.chip
        self.g_pot += data.chip
        if data.uid in self.dict_money_on_table.keys():
            self.dict_money_on_table[data.uid] += data.chip
        else:
            self.dict_money_on_table[data.uid] = data.chip

    # 有玩家下注
    async def on_belt(self, data):
        operation = {'uid': data.uid, 'op':Op.Belt, 'chip': data.chip}
        self.action_done.append(operation)

        logger.info('on_belt:{}', operation)

        if data.uid == self.my_uid:
            self.available_chip -= data.chip
            self.money_on_the_table += data.chip
        self.g_pot += data.chip
        if data.uid in self.dict_money_on_table.keys():
            self.dict_money_on_table[data.uid] += data.chip
        else:
            self.dict_money_on_table[data.uid] = data.chip

    # 有玩家加注
    async def on_fill(self, data):
        operation = {'uid': data.uid, 'op':Op.Raise, 'chip': data.chip}
        self.action_done.append(operation)

        logger.info('on_fill:{}', operation)

        if data.uid == self.my_uid:
            self.available_chip -= data.chip
            self.money_on_the_table += data.chip
        self.g_pot += data.chip
        if data.uid in self.dict_money_on_table.keys():
            self.dict_money_on_table[data.uid] += data.chip
        else:
            self.dict_money_on_table[data.uid] = data.chip

    # 有玩家ALL IN
    async def on_all_in(self, data):
        operation = {'uid': data.uid, 'op':Op.AllIn, 'chip': data.chip}
        self.action_done.append(operation)

        logger.info('on_all_in:{}', operation)

        if data.uid == self.my_uid:
            self.available_chip -= data.chip
            self.money_on_the_table += data.chip
        g_pot += data.chip
        if data.uid in self.dict_money_on_table.keys():
            self.dict_money_on_table[data.uid] += data.chip
        else:
            self.dict_money_on_table[data.uid] = data.chip

    async def on_add_chips_suc(self, data):
        if data.uid == self.my_uid:
            self.available_chip += data.chip

    async def on_arrange_players_chip(self, data):
        players = data.players
        for player in players:
            if self.my_uid == player.uid:
                if player.chip and player.chip > 0:
                    self.available_chip += player.chip

    async def on_diss_card(self, data):
        self.g_player_num -= 1
        logger.info('on_diss_card:{}', data.uid)

    # 断线重连
    async def on_room_snapshot(self, data):
        self.my_hand_cards = []
        players = data.players
        total_cur_belt_chip = 0
        for player in players:
            if player.isPlaying:
                self.g_player_num += 1
            if player.uid != self.my_uid:
                self.available_chip = player.chip
            else:
                pass
            self.dict_money_on_table[player.uid] = player.currentTotalChip
        g_pot = data.pot
        for card in data.handCards:
            self.my_hand_cards.append(convert_card(card))
        for card in data.tableCards:
            self.table_cards.append(convert_card(card))

        await self.choice_operation(data.opButtons)

    async def on_enter_my_room(self, data):
        logger.info('on_enter_my_room--------------------{}', data)
        if data.errorCode != 0:
            # logger.info(data.errorMsg)
            await self.send_enter_room()
        else:
            pass


async def single_client(player_id):
    client = WebSocketClient()
    try:
        await client.connect()
        logger.info("Player {} sending login...", player_id)
        # 设置要进入的房间号密码
        client.set_room_no_pwd("212810093", "0989")
        # 发送登录请求,第一个参数“用户名”，第二个参数“密码”，没有账号账号自动创建
        await client.send_login_msg(f"player{player_id:02d}", "123")
        
        while client._running:
            await asyncio.sleep(1)
    finally:
        # logger.info("leave--------------------------", player_id)
        await client.disconnect()

async def main():
    # 同时启动50个客户端
    clients = [single_client(i) for i in range(1, 2)]
    await asyncio.gather(*clients)  # 并发运行所有客户端

if __name__ == "__main__":
    asyncio.run(main())