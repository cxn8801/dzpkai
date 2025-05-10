# def get_player_position(seat_dict, dealer_uid, player_uid):
#     """
#     :param seat_dict: dict {seat_number: uid}
#     :param dealer_uid: int  # 庄家uid
#     :param player_uid: int  # 需要查询的玩家uid
#     :return: str  # 位置名
#     """
#     pos_names = ['SB', 'BB', 'UTG', 'MP', 'CO']  # BTN单独处理
#     occupied_seats = sorted(seat_dict.keys())
#     num_players = len(occupied_seats)
#     if num_players < 2:
#         return None

#     # 找到庄家的座位号
#     dealer_seat = None
#     for seat_num, uid in seat_dict.items():
#         if uid == dealer_uid:
#             dealer_seat = seat_num
#             break
#     if dealer_seat is None:
#         return None

#     # 按顺时针，从庄家seat开始排列
#     dealer_idx = occupied_seats.index(dealer_seat)
#     ordered_seats = occupied_seats[dealer_idx:] + occupied_seats[:dealer_idx]
#     ordered_uids = [seat_dict[seat] for seat in ordered_seats]

#     # BTN是庄家
#     btn_uid = ordered_uids[0]
#     # 其余的按顺时针（SB、BB、UTG、MP、CO...），不足按实际人数分配
#     positions = ['BTN']
#     # n个玩家时，BTN后面依次分配SB, BB, UTG, MP, CO, MP...
#     for i in range(1, num_players):
#         if i - 1 < len(pos_names):
#             positions.append(pos_names[i-1])
#         else:
#             positions.append('MP')

#     # 查询玩家位置
#     try:
#         idx = ordered_uids.index(player_uid)
#     except ValueError:
#         return None
#     return positions[idx]


def get_player_position(seat_dict, dealer_uid, target_uid):
    # 找到按钮座位
    btn_seat = None
    for seat, uid in seat_dict.items():
        if uid == dealer_uid:
            btn_seat = seat
            break
    if btn_seat is None:
        return None  # 庄家UID不存在于座位字典中，视为无效输入
    
    # 生成按钮后的顺时针顺序
    def get_clockwise_order(start_seat):
        order = []
        current = start_seat % 9 + 1
        for _ in range(9):
            order.append(current)
            current = current % 9 + 1
        return order
    
    # 寻找SB的座位
    order_after_btn = get_clockwise_order(btn_seat)
    sb_seat = None
    for seat in order_after_btn:
        if seat in seat_dict and seat != btn_seat:
            sb_seat = seat
            break
    if sb_seat is None:
        return None  # 没有找到SB，但根据题意玩家至少两人，应该不会发生
    
    # 寻找BB的座位
    order_after_sb = get_clockwise_order(sb_seat)
    bb_seat = None
    for seat in order_after_sb:
        if seat in seat_dict and seat != sb_seat and seat != btn_seat:
            bb_seat = seat
            break
    if bb_seat is None:
        return None  # 没有找到BB，同样根据题意应该不会发生
    
    # 确定剩下的玩家顺序，从BB之后开始
    order_after_bb = get_clockwise_order(bb_seat)
    remaining_players = []
    for seat in order_after_bb:
        if seat in seat_dict and seat not in {btn_seat, sb_seat, bb_seat}:
            remaining_players.append(seat)
    
    # 构建位置映射
    position_map = {
        btn_seat: 'BTN',
        sb_seat: 'SB',
        bb_seat: 'BB'
    }
    
    # 分配UTG, MP, CO
    positions = ['UTG', 'MP', 'CO']
    for idx, seat in enumerate(remaining_players[:3]):
        position_map[seat] = positions[idx]
    
    # 查找目标玩家的座位
    target_seat = None
    for seat, uid in seat_dict.items():
        if uid == target_uid:
            target_seat = seat
            break
    if target_seat is None:
        return None  # 目标UID不存在
    
    return position_map.get(target_seat, None)

# 用户示例测试
if __name__ == "__main__":
    seat_dict = {
        1: 5,
        2: 6,
        3: 102, # bank id 庄家
        5: 103,
        6: 55,
        8: 77
    }
    print(get_player_position(seat_dict, 102, 5))    # MP
    print(get_player_position(seat_dict, 102, 6))    # CO
    print(get_player_position(seat_dict, 102, 102))  # BTN
    print(get_player_position(seat_dict, 102, 103))  # SB
    print(get_player_position(seat_dict, 102, 55))   # BB
    print(get_player_position(seat_dict, 102, 77))   # UTG