# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: dzpk.proto
# Protobuf Python Version: 4.25.2
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\ndzpk.proto\x12\x0b\x63sdzpkproto\"\xfc\x02\n\nRoomPlayer\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x12\n\nseatNumber\x18\x02 \x01(\r\x12\x0c\n\x04name\x18\x03 \x01(\t\x12\x0c\n\x04head\x18\x04 \x01(\t\x12\x0c\n\x04gold\x18\x05 \x01(\x03\x12\x12\n\nisPrepared\x18\x06 \x01(\x08\x12\x0e\n\x06gender\x18\x07 \x01(\x05\x12\r\n\x05score\x18\x08 \x01(\x03\x12\x10\n\x08takeChip\x18\t \x01(\x03\x12\x12\n\nisHoldDesk\x18\n \x01(\x08\x12\x14\n\x0cisHoldDesked\x18\x0b \x01(\x08\x12\x10\n\x08interval\x18\x0c \x01(\x04\x12\x0e\n\x06stopAt\x18\r \x01(\x04\x12\x14\n\x0cisPopAddChip\x18\x0e \x01(\x08\x12\x11\n\tOpEndTime\x18\x0f \x01(\x03\x12\r\n\x05\x63\x61rds\x18\x10 \x03(\x05\x12\x0f\n\x07\x61llChip\x18\x11 \x01(\x04\x12\x0c\n\x04\x63hip\x18\x12 \x01(\x03\x12\x18\n\x10\x63urrentTotalChip\x18\x13 \x01(\x04\x12\x0e\n\x06status\x18\x14 \x01(\x05\x12\x11\n\tisPlaying\x18\x15 \x01(\x08\"\x87\x01\n\nPlayerInfo\x12\x0e\n\x06Status\x18\x01 \x01(\x05\x12\x0f\n\x07WinChip\x18\x02 \x01(\x03\x12\x11\n\tHandCards\x18\x03 \x03(\x05\x12\r\n\x05Level\x18\x04 \x01(\x05\x12\r\n\x05\x43\x61rds\x18\x05 \x03(\x05\x12\x0c\n\x04name\x18\x06 \x01(\t\x12\x0c\n\x04head\x18\x07 \x01(\t\x12\x0b\n\x03uid\x18\x08 \x01(\x03\"-\n\rplayerBetInfo\x12\x0e\n\x06Status\x18\x01 \x01(\x05\x12\x0c\n\x04\x63hip\x18\x02 \x01(\x03\" \n\x0b\x43SEnterRoom\x12\x11\n\troomLevel\x18\x01 \x01(\r\"_\n\x0bSCEnterRoom\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12+\n\nroomPlayer\x18\x03 \x01(\x0b\x32\x17.csdzpkproto.RoomPlayer\"9\n\x12\x43SCreateFriendRoom\x12\x11\n\tbingBlind\x18\x01 \x01(\x05\x12\x10\n\x08overTime\x18\x02 \x01(\x05\"5\n\x11\x43SEnterFriendRoom\x12\x0e\n\x06roomNo\x18\x01 \x01(\t\x12\x10\n\x08password\x18\x02 \x01(\t\"\x15\n\x13\x43SRoomScenePrepared\"9\n\x0bReplayEntry\x12\x13\n\x0bmessageName\x18\x01 \x01(\t\x12\x15\n\rmessageStream\x18\x02 \x01(\x0c\"\x19\n\x04Rule\x12\x11\n\tbaseScore\x18\x01 \x01(\r\"\xba\x03\n\x0eSCRoomSnapshot\x12(\n\x07players\x18\x01 \x03(\x0b\x32\x17.csdzpkproto.RoomPlayer\x12/\n\rreplayEntries\x18\x02 \x03(\x0b\x32\x18.csdzpkproto.ReplayEntry\x12\x1f\n\x04rule\x18\x03 \x01(\x0b\x32\x11.csdzpkproto.Rule\x12\x11\n\troomLevel\x18\x04 \x01(\r\x12\x0f\n\x07isMatch\x18\x05 \x01(\x08\x12\x0f\n\x07matchId\x18\x06 \x01(\x04\x12\x15\n\rsimpleMatchId\x18\x07 \x01(\r\x12\x11\n\tmatchName\x18\x08 \x01(\t\x12\x14\n\x0cmatchBegTime\x18\t \x01(\x04\x12\x12\n\ntableCards\x18\n \x03(\x05\x12\x11\n\thandCards\x18\x0b \x03(\x05\x12*\n\topButtons\x18\x0c \x03(\x0b\x32\x17.csdzpkproto.OpDzButton\x12\x15\n\rbankerSeatNum\x18\r \x01(\r\x12\x0b\n\x03pot\x18\x0e \x01(\x04\x12\x0e\n\x06roomId\x18\x0f \x01(\x04\x12\x10\n\x08password\x18\x10 \x01(\t\x12\x0e\n\x06roomNo\x18\x11 \x01(\t\x12\x0e\n\x06manage\x18\x12 \x01(\x08\"\x1f\n\x0b\x43SLeaveRoom\x12\x10\n\x08isForced\x18\x01 \x01(\x08\"d\n\x0bSCLeaveRoom\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\x12\x11\n\tisRemoved\x18\x04 \x01(\x08\x12\x10\n\x08roomType\x18\x05 \x01(\x05\"\x0b\n\tCSDismiss\"0\n\tSCDismiss\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\"\n\n\x08\x43SManage\"<\n\x08SCManage\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\"\x10\n\x0e\x43SCancelManage\"B\n\x0eSCCancelManage\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\"\x18\n\tSCOffline\x12\x0b\n\x03uid\x18\x01 \x01(\r\"\x1f\n\x06\x43SChat\x12\x15\n\rchatMessageId\x18\x01 \x01(\r\"Q\n\x06SCChat\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\x12\x15\n\rchatMessageId\x18\x04 \x01(\r\".\n\tCSUseItem\x12\x0e\n\x06itemId\x18\x01 \x01(\r\x12\x11\n\ttargetUid\x18\x02 \x01(\r\"n\n\tSCUseItem\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\x12\x0e\n\x06itemId\x18\x04 \x01(\r\x12\x11\n\ttargetUid\x18\x05 \x01(\r\x12\x0c\n\x04gold\x18\x06 \x01(\x03\"%\n\x13SCStartPrepareTimer\x12\x0e\n\x06stopAt\x18\x01 \x01(\x04\"\x14\n\x12SCStopPrepareTimer\"\x0b\n\tCSPrepare\"=\n\tSCPrepare\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\"E\n\x0bSCGameStart\x12\x10\n\x08\x62\x61nkerId\x18\x01 \x01(\r\x12\x12\n\nSmallBlind\x18\x02 \x01(\x04\x12\x10\n\x08\x42igBlind\x18\x03 \x01(\x04\"\x16\n\x14SCStartDeterminePiao\" \n\x0f\x43SDeterminePiao\x12\r\n\x05times\x18\x01 \x01(\r\"R\n\x0fSCDeterminePiao\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\x12\r\n\x05times\x18\x04 \x01(\r\"E\n\x0bPlayerCards\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x11\n\thandCards\x18\x02 \x03(\x05\x12\x16\n\x0ehandCardNumber\x18\x03 \x01(\r\"b\n\x14SCDispatchStartCards\x12-\n\x0bplayerCards\x18\x01 \x03(\x0b\x32\x18.csdzpkproto.PlayerCards\x12\x1b\n\x13tableLeftCardNumber\x18\x02 \x01(\r\"~\n\x12\x43omparePlayerCards\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x11\n\thandCards\x18\x02 \x03(\x05\x12\x13\n\x0b\x63hooseCards\x18\x03 \x03(\x05\x12\x11\n\tleftCards\x18\x04 \x03(\x05\x12\x11\n\tfiveCards\x18\x05 \x03(\x05\x12\r\n\x05level\x18\x06 \x01(\x05\"W\n\x0eSCCompareCards\x12\x34\n\x0bplayerCards\x18\x01 \x03(\x0b\x32\x1f.csdzpkproto.ComparePlayerCards\x12\x0f\n\x07uidList\x18\x02 \x03(\r\"$\n\x07Players\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x0c\n\x04\x63hip\x18\x02 \x01(\x03\"N\n\x14SCArrangePlayersChip\x12%\n\x07players\x18\x01 \x03(\x0b\x32\x14.csdzpkproto.Players\x12\x0f\n\x07uidList\x18\x02 \x03(\r\"?\n\x16SCPlayerLoseAndWinInfo\x12%\n\x07players\x18\x01 \x03(\x0b\x32\x14.csdzpkproto.Players\"\x1e\n\x0fSCUpdateOpLight\x12\x0b\n\x03uid\x18\x01 \x01(\r\"\x11\n\x0fSCSuspendCardOp\"\x19\n\tCSOutCard\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\"K\n\tSCOutCard\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\x12\x0c\n\x04\x63\x61rd\x18\x04 \x01(\x05\"&\n\tSCGetCard\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x0c\n\x04\x63\x61rd\x18\x02 \x01(\x05\"H\n\x08SCOpList\x12(\n\topButtons\x18\x02 \x03(\x0b\x32\x15.csdzpkproto.OpButton\x12\x12\n\ncanOutCard\x18\x03 \x01(\x08\"8\n\nSCDzOpList\x12*\n\topButtons\x18\x02 \x03(\x0b\x32\x17.csdzpkproto.OpDzButton\"G\n\x08OpButton\x12\x10\n\x08\x62uttonId\x18\x01 \x01(\r\x12)\n\topCtxList\x18\x02 \x03(\x0b\x32\x16.csdzpkproto.OpContext\"=\n\nOpDzButton\x12\x10\n\x08\x62uttonId\x18\x01 \x01(\r\x12\x0c\n\x04\x63hip\x18\x02 \x01(\x04\x12\x0f\n\x07maxChip\x18\x03 \x01(\x04\"\x96\x02\n\tOpContext\x12\x0c\n\x04opId\x18\x01 \x01(\r\x12+\n\x08\x63hiOpCtx\x18\x02 \x01(\x0b\x32\x19.csdzpkproto.ChiOpContext\x12-\n\tpengOpCtx\x18\x03 \x01(\x0b\x32\x1a.csdzpkproto.PengOpContext\x12\x35\n\rmingGangOpCtx\x18\x04 \x01(\x0b\x32\x1e.csdzpkproto.MingGangOpContext\x12\x31\n\x0b\x61nGangOpCtx\x18\x05 \x01(\x0b\x32\x1c.csdzpkproto.AnGangOpContext\x12\x35\n\rpengGangOpCtx\x18\x06 \x01(\x0b\x32\x1e.csdzpkproto.PengGangOpContext\"\x98\x02\n\x0bOpDzContext\x12\x0c\n\x04opId\x18\x01 \x01(\r\x12+\n\x08\x63hiOpCtx\x18\x02 \x01(\x0b\x32\x19.csdzpkproto.ChiOpContext\x12-\n\tpengOpCtx\x18\x03 \x01(\x0b\x32\x1a.csdzpkproto.PengOpContext\x12\x35\n\rmingGangOpCtx\x18\x04 \x01(\x0b\x32\x1e.csdzpkproto.MingGangOpContext\x12\x31\n\x0b\x61nGangOpCtx\x18\x05 \x01(\x0b\x32\x1c.csdzpkproto.AnGangOpContext\x12\x35\n\rpengGangOpCtx\x18\x06 \x01(\x0b\x32\x1e.csdzpkproto.PengGangOpContext\"5\n\x0c\x43hiOpContext\x12\x11\n\tusedCards\x18\x01 \x03(\x05\x12\x12\n\ntargetCard\x18\x02 \x01(\x05\"\x1d\n\rPengOpContext\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\"!\n\x11MingGangOpContext\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\"\x1f\n\x0f\x41nGangOpContext\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\"!\n\x11PengGangOpContext\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\"\x08\n\x06\x43SPass\"-\n\x06SCPass\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\".\n\x05\x43SChi\x12\x11\n\tusedCards\x18\x01 \x03(\x05\x12\x12\n\ntargetCard\x18\x02 \x01(\x05\"s\n\x05SCChi\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\x12\x11\n\tusedCards\x18\x04 \x03(\x05\x12\x12\n\ntargetCard\x18\x05 \x01(\x05\x12\x11\n\ttargetUid\x18\x06 \x01(\r\"\x16\n\x06\x43SPeng\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\"[\n\x06SCPeng\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\x12\x0c\n\x04\x63\x61rd\x18\x04 \x01(\x05\x12\x11\n\ttargetUid\x18\x05 \x01(\r\".\n\nCSMingGang\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\x12\x12\n\nisRollDice\x18\x02 \x01(\x08\"s\n\nSCMingGang\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\x12\x0c\n\x04\x63\x61rd\x18\x04 \x01(\x05\x12\x12\n\nisRollDice\x18\x05 \x01(\x08\x12\x11\n\ttargetUid\x18\x06 \x01(\r\",\n\x08\x43SAnGang\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\x12\x12\n\nisRollDice\x18\x02 \x01(\x08\"^\n\x08SCAnGang\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\x12\x0c\n\x04\x63\x61rd\x18\x04 \x01(\x05\x12\x12\n\nisRollDice\x18\x05 \x01(\x08\".\n\nCSPengGang\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\x12\x12\n\nisRollDice\x18\x02 \x01(\x08\"`\n\nSCPengGang\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x0b\n\x03uid\x18\x03 \x01(\r\x12\x0c\n\x04\x63\x61rd\x18\x04 \x01(\x05\x12\x12\n\nisRollDice\x18\x05 \x01(\x08\"D\n\x12SCGetDiceGangCards\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\r\n\x05\x63\x61rds\x18\x02 \x03(\x05\x12\x12\n\ncardNumber\x18\x03 \x01(\r\"3\n\x15SCGiveUpDiceGangCards\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\r\n\x05\x63\x61rds\x18\x02 \x03(\x05\"*\n\x19SCWaitOpWithDiceGangCards\x12\r\n\x05\x63\x61rds\x18\x01 \x03(\x05\"\x1f\n\x1dSCStopWaitOpWithDiceGangCards\"\x06\n\x04\x43SHu\"\xc7\x01\n\x04SCHu\x12\x11\n\terrorCode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x32\n\x0bplayerCards\x18\x04 \x03(\x0b\x32\x1d.csdzpkproto.SCHu.PlayerCards\x12\x12\n\ndianpaoUid\x18\x05 \x01(\r\x12\x15\n\rdiceGangCards\x18\x06 \x03(\x05\x1a;\n\x0bPlayerCards\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x11\n\thandCards\x18\x02 \x03(\x05\x12\x0c\n\x04isHu\x18\x03 \x01(\x08\"\x88\x02\n\x0bTingpaiData\x12/\n\x07\x65ntries\x18\x01 \x03(\x0b\x32\x1e.csdzpkproto.TingpaiData.Entry\x12?\n\x0ftingCardEntries\x18\x03 \x03(\x0b\x32&.csdzpkproto.TingpaiData.TingCardEntry\x1aY\n\x05\x45ntry\x12\x0f\n\x07outCard\x18\x01 \x01(\x05\x12?\n\x0ftingCardEntries\x18\x03 \x03(\x0b\x32&.csdzpkproto.TingpaiData.TingCardEntry\x1a,\n\rTingCardEntry\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\x12\r\n\x05times\x18\x02 \x01(\x05\">\n\rSCTingpaiData\x12-\n\x0btingpaiData\x18\x01 \x01(\x0b\x32\x18.csdzpkproto.TingpaiData\"s\n\tSCZhaniao\x12-\n\x07\x65ntries\x18\x01 \x03(\x0b\x32\x1c.csdzpkproto.SCZhaniao.Entry\x1a\x37\n\x05\x45ntry\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\x12\x11\n\ttargetUid\x18\x02 \x01(\r\x12\r\n\x05isHit\x18\x03 \x01(\x08\"\x17\n\x08SCManyou\x12\x0b\n\x03uid\x18\x01 \x01(\r\"\x10\n\x0e\x43SGetHaidiCard\"+\n\x0eSCGetHaidiCard\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x0c\n\x04\x63\x61rd\x18\x02 \x01(\x05\"_\n\tSCStartHu\x12-\n\x07\x65ntries\x18\x01 \x03(\x0b\x32\x1c.csdzpkproto.SCStartHu.Entry\x1a#\n\x05\x45ntry\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\r\n\x05\x63\x61rds\x18\x02 \x03(\x05\"\x15\n\x13SCWaitOpWithStartHu\"\x19\n\x17SCStopWaitOpWithStartHu\"\t\n\x07SCLiuju\"\x94\x01\n\x13JiesuanStartHuBlock\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x39\n\x07\x65ntries\x18\x02 \x03(\x0b\x32(.csdzpkproto.JiesuanStartHuBlock.HuEntry\x12\r\n\x05times\x18\x03 \x01(\x05\x12\r\n\x05score\x18\x04 \x01(\x03\x1a\x17\n\x07HuEntry\x12\x0c\n\x04type\x18\x01 \x01(\r\"r\n\x0eJiesuanHuBlock\x12\x0e\n\x06isZimo\x18\x01 \x01(\x08\x12\r\n\x05huUid\x18\x02 \x01(\r\x12\x12\n\ndianpaoUid\x18\x03 \x01(\r\x12\x0f\n\x07huTypes\x18\x04 \x03(\r\x12\r\n\x05times\x18\x05 \x01(\x05\x12\r\n\x05score\x18\x06 \x01(\x03\"0\n\x10JiesuanPiaoBlock\x12\r\n\x05times\x18\x01 \x01(\x05\x12\r\n\x05score\x18\x02 \x01(\x03\"$\n\x13JiesuanZhaniaoBlock\x12\r\n\x05times\x18\x01 \x01(\x05\"\xe8\x02\n\rJiesuanPlayer\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0c\n\x04head\x18\x03 \x01(\t\x12\x12\n\nseatNumber\x18\x04 \x01(\r\x12\r\n\x05score\x18\x05 \x01(\x03\x12\x13\n\x0bisTopWinner\x18\x06 \x01(\x08\x12\x12\n\nisBankrupt\x18\x07 \x01(\x08\x12\x10\n\x08isBanker\x18\x08 \x01(\x08\x12\x36\n\x0czhaniaoBlock\x18\t \x01(\x0b\x32 .csdzpkproto.JiesuanZhaniaoBlock\x12\x37\n\rstartHuBlocks\x18\x0c \x03(\x0b\x32 .csdzpkproto.JiesuanStartHuBlock\x12-\n\x08huBlocks\x18\r \x03(\x0b\x32\x1b.csdzpkproto.JiesuanHuBlock\x12\x30\n\tpiaoBlock\x18\x0e \x01(\x0b\x32\x1d.csdzpkproto.JiesuanPiaoBlock\"\xb7\x01\n\tSCJiesuan\x12\x14\n\x0czhaniaoCards\x18\x01 \x03(\x05\x12+\n\x07players\x18\x02 \x03(\x0b\x32\x1a.csdzpkproto.JiesuanPlayer\x12\x0f\n\x07isMatch\x18\x03 \x01(\x08\x12\x19\n\x11isMatchFinalStage\x18\x04 \x01(\x08\x12\x1c\n\x14matchStageRoundIndex\x18\x05 \x01(\r\x12\x1d\n\x15isMatchStageLastRound\x18\x06 \x01(\x08\"\x15\n\x13\x43SContinueNextRound\"\x13\n\x04\x43SGm\x12\x0b\n\x03\x63md\x18\x01 \x01(\t\"=\n\x0cSCStartTimer\x12\x10\n\x08interval\x18\x01 \x01(\x04\x12\x0b\n\x03uid\x18\x02 \x01(\r\x12\x0e\n\x06stopAt\x18\x03 \x01(\x04\"\r\n\x0bSCStopTimer\"\x1a\n\x18\x43SRoundJiesuanViewClosed\"2\n\x07SCBlind\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x0c\n\x04\x63hip\x18\x02 \x01(\x03\x12\x0c\n\x04kind\x18\x03 \x01(\r\"\x16\n\x06\x43SBelt\x12\x0c\n\x04\x63hip\x18\x02 \x01(\x03\"3\n\x06SCBelt\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x0c\n\x04\x63hip\x18\x02 \x01(\x03\x12\x0e\n\x06status\x18\x03 \x01(\r\"\x16\n\x06\x43SCall\x12\x0c\n\x04\x63hip\x18\x02 \x01(\x03\"3\n\x06SCCall\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x0c\n\x04\x63hip\x18\x02 \x01(\x03\x12\x0e\n\x06status\x18\x03 \x01(\r\"\x16\n\x06\x43SFill\x12\x0c\n\x04\x63hip\x18\x02 \x01(\x03\"3\n\x06SCFill\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x0c\n\x04\x63hip\x18\x02 \x01(\x03\x12\x0e\n\x06status\x18\x03 \x01(\r\"\x0c\n\nCSDissCard\")\n\nSCDissCard\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x0e\n\x06status\x18\x03 \x01(\r\"\x0c\n\nCSLookCard\")\n\nSCLookCard\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x0e\n\x06status\x18\x03 \x01(\r\"$\n\x07\x43SAllIn\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x0c\n\x04\x63hip\x18\x02 \x01(\x03\"4\n\x07SCAllIn\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x0c\n\x04\x63hip\x18\x02 \x01(\x03\x12\x0e\n\x06status\x18\x03 \x01(\r\"\x10\n\x0eSCStartPreFlop\"\x1c\n\x0bSCStartFlop\x12\r\n\x05\x63\x61rds\x18\x01 \x03(\x05\"\x1b\n\x0bSCStartTurn\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\"\x1c\n\x0cSCStartRiver\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\x05\"\x11\n\x0fSCStartShowDown\",\n\tCSSitDown\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x12\n\nseatNumber\x18\x02 \x01(\r\">\n\tSCSitDown\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x12\n\nseatNumber\x18\x02 \x01(\r\x12\x10\n\x08isLooked\x18\x03 \x01(\x08\" \n\nCSHoldDesk\x12\x12\n\nseatNumber\x18\x02 \x01(\r\"O\n\nSCHoldDesk\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x12\n\nseatNumber\x18\x02 \x01(\r\x12\x10\n\x08interval\x18\x03 \x01(\x04\x12\x0e\n\x06stopAt\x18\x04 \x01(\x04\"\x1f\n\tCSStandUp\x12\x12\n\nseatNumber\x18\x02 \x01(\r\",\n\tSCStandUp\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x12\n\nseatNumber\x18\x02 \x01(\r\"4\n\nCSAddChips\x12\x12\n\nplussMoney\x18\x02 \x01(\x04\x12\x12\n\nisPlussMax\x18\x03 \x01(\x08\"\x0f\n\rCSCanCelChips\"\x0e\n\x0c\x43SPlussChips\"Z\n\nSCAddChips\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x11\n\tleftMoney\x18\x02 \x01(\x04\x12\x0b\n\x03min\x18\x03 \x01(\x04\x12\x0b\n\x03max\x18\x04 \x01(\x04\x12\x12\n\nisPlussMax\x18\x05 \x01(\x08\"\x0c\n\nCSBackGame\"\x19\n\nSCBackGame\x12\x0b\n\x03uid\x18\x01 \x01(\r\"\x1f\n\x10\x43SCancelHoldDesk\x12\x0b\n\x03uid\x18\x01 \x01(\r\"\x1f\n\x10SCCancelHoldDesk\x12\x0b\n\x03uid\x18\x01 \x01(\r\"0\n\rSCHoldDeskSuc\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x12\n\nisJoinGame\x18\x02 \x01(\x08\"\x1e\n\x0fSCCancleDeskSuc\x12\x0b\n\x03uid\x18\x01 \x01(\r\">\n\rSCAddChipsSuc\x12\x0b\n\x03uid\x18\x01 \x01(\r\x12\x12\n\nisJoinGame\x18\x02 \x01(\x08\x12\x0c\n\x04\x63hip\x18\x03 \x01(\x04\"&\n\x16\x43SRoomPlayerHandRecord\x12\x0c\n\x04Page\x18\x01 \x01(\r\"y\n\x16SCRoomPlayerHandRecord\x12\x11\n\ttotalPage\x18\x01 \x01(\x05\x12-\n\x0cPlayListInfo\x18\x02 \x03(\x0b\x32\x17.csdzpkproto.PlayerInfo\x12\x1d\n\x15isFirstOpenHandRecord\x18\x03 \x01(\x08\"h\n\x14SCMacthRoomBlindInfo\x12\x12\n\nsmallBlind\x18\x01 \x01(\x05\x12\x0c\n\x04\x61nte\x18\x02 \x01(\x05\x12\x13\n\x0brefreshTime\x18\x03 \x01(\x03\x12\x19\n\x11matchRoomBlindNum\x18\x04 \x01(\r\"/\n\x17SCMacthRoomIsFinalTable\x12\x14\n\x0cisFinalTable\x18\x01 \x01(\x08\"D\n\x15SCMatchRoomRepeatInfo\x12\x11\n\tcountdown\x18\x01 \x01(\x05\x12\x18\n\x10playerResidueNum\x18\x02 \x01(\x05\"H\n\x12SCClickOnTheRepeat\x12\x0c\n\x04\x63ode\x18\x01 \x01(\x05\x12\x10\n\x08\x65rrorMsg\x18\x02 \x01(\t\x12\x12\n\nplayerChip\x18\x03 \x01(\x04\"\x14\n\x12\x43SClickOnTheRepeat\"*\n\x17SCRoomPlayerOpCountdown\x12\x0f\n\x07\x65ndTime\x18\x01 \x01(\x03\"\r\n\x0bSCstartGame\"B\n\x16\x43SShowHandCardListInfo\x12(\n\x05\x63\x61rds\x18\x01 \x03(\x0b\x32\x19.csdzpkproto.ShowHandCard\",\n\x0cShowHandCard\x12\x0c\n\x04\x63\x61rd\x18\x01 \x01(\r\x12\x0e\n\x06status\x18\x02 \x01(\r\"0\n\rSCSetShowCard\x12\x0f\n\x07\x65rrCode\x18\x01 \x01(\r\x12\x0e\n\x06\x65rrMsg\x18\x02 \x01(\t\"\x13\n\x11SCRequestHandCardb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'dzpk_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_ROOMPLAYER']._serialized_start=28
  _globals['_ROOMPLAYER']._serialized_end=408
  _globals['_PLAYERINFO']._serialized_start=411
  _globals['_PLAYERINFO']._serialized_end=546
  _globals['_PLAYERBETINFO']._serialized_start=548
  _globals['_PLAYERBETINFO']._serialized_end=593
  _globals['_CSENTERROOM']._serialized_start=595
  _globals['_CSENTERROOM']._serialized_end=627
  _globals['_SCENTERROOM']._serialized_start=629
  _globals['_SCENTERROOM']._serialized_end=724
  _globals['_CSCREATEFRIENDROOM']._serialized_start=726
  _globals['_CSCREATEFRIENDROOM']._serialized_end=783
  _globals['_CSENTERFRIENDROOM']._serialized_start=785
  _globals['_CSENTERFRIENDROOM']._serialized_end=838
  _globals['_CSROOMSCENEPREPARED']._serialized_start=840
  _globals['_CSROOMSCENEPREPARED']._serialized_end=861
  _globals['_REPLAYENTRY']._serialized_start=863
  _globals['_REPLAYENTRY']._serialized_end=920
  _globals['_RULE']._serialized_start=922
  _globals['_RULE']._serialized_end=947
  _globals['_SCROOMSNAPSHOT']._serialized_start=950
  _globals['_SCROOMSNAPSHOT']._serialized_end=1392
  _globals['_CSLEAVEROOM']._serialized_start=1394
  _globals['_CSLEAVEROOM']._serialized_end=1425
  _globals['_SCLEAVEROOM']._serialized_start=1427
  _globals['_SCLEAVEROOM']._serialized_end=1527
  _globals['_CSDISMISS']._serialized_start=1529
  _globals['_CSDISMISS']._serialized_end=1540
  _globals['_SCDISMISS']._serialized_start=1542
  _globals['_SCDISMISS']._serialized_end=1590
  _globals['_CSMANAGE']._serialized_start=1592
  _globals['_CSMANAGE']._serialized_end=1602
  _globals['_SCMANAGE']._serialized_start=1604
  _globals['_SCMANAGE']._serialized_end=1664
  _globals['_CSCANCELMANAGE']._serialized_start=1666
  _globals['_CSCANCELMANAGE']._serialized_end=1682
  _globals['_SCCANCELMANAGE']._serialized_start=1684
  _globals['_SCCANCELMANAGE']._serialized_end=1750
  _globals['_SCOFFLINE']._serialized_start=1752
  _globals['_SCOFFLINE']._serialized_end=1776
  _globals['_CSCHAT']._serialized_start=1778
  _globals['_CSCHAT']._serialized_end=1809
  _globals['_SCCHAT']._serialized_start=1811
  _globals['_SCCHAT']._serialized_end=1892
  _globals['_CSUSEITEM']._serialized_start=1894
  _globals['_CSUSEITEM']._serialized_end=1940
  _globals['_SCUSEITEM']._serialized_start=1942
  _globals['_SCUSEITEM']._serialized_end=2052
  _globals['_SCSTARTPREPARETIMER']._serialized_start=2054
  _globals['_SCSTARTPREPARETIMER']._serialized_end=2091
  _globals['_SCSTOPPREPARETIMER']._serialized_start=2093
  _globals['_SCSTOPPREPARETIMER']._serialized_end=2113
  _globals['_CSPREPARE']._serialized_start=2115
  _globals['_CSPREPARE']._serialized_end=2126
  _globals['_SCPREPARE']._serialized_start=2128
  _globals['_SCPREPARE']._serialized_end=2189
  _globals['_SCGAMESTART']._serialized_start=2191
  _globals['_SCGAMESTART']._serialized_end=2260
  _globals['_SCSTARTDETERMINEPIAO']._serialized_start=2262
  _globals['_SCSTARTDETERMINEPIAO']._serialized_end=2284
  _globals['_CSDETERMINEPIAO']._serialized_start=2286
  _globals['_CSDETERMINEPIAO']._serialized_end=2318
  _globals['_SCDETERMINEPIAO']._serialized_start=2320
  _globals['_SCDETERMINEPIAO']._serialized_end=2402
  _globals['_PLAYERCARDS']._serialized_start=2404
  _globals['_PLAYERCARDS']._serialized_end=2473
  _globals['_SCDISPATCHSTARTCARDS']._serialized_start=2475
  _globals['_SCDISPATCHSTARTCARDS']._serialized_end=2573
  _globals['_COMPAREPLAYERCARDS']._serialized_start=2575
  _globals['_COMPAREPLAYERCARDS']._serialized_end=2701
  _globals['_SCCOMPARECARDS']._serialized_start=2703
  _globals['_SCCOMPARECARDS']._serialized_end=2790
  _globals['_PLAYERS']._serialized_start=2792
  _globals['_PLAYERS']._serialized_end=2828
  _globals['_SCARRANGEPLAYERSCHIP']._serialized_start=2830
  _globals['_SCARRANGEPLAYERSCHIP']._serialized_end=2908
  _globals['_SCPLAYERLOSEANDWININFO']._serialized_start=2910
  _globals['_SCPLAYERLOSEANDWININFO']._serialized_end=2973
  _globals['_SCUPDATEOPLIGHT']._serialized_start=2975
  _globals['_SCUPDATEOPLIGHT']._serialized_end=3005
  _globals['_SCSUSPENDCARDOP']._serialized_start=3007
  _globals['_SCSUSPENDCARDOP']._serialized_end=3024
  _globals['_CSOUTCARD']._serialized_start=3026
  _globals['_CSOUTCARD']._serialized_end=3051
  _globals['_SCOUTCARD']._serialized_start=3053
  _globals['_SCOUTCARD']._serialized_end=3128
  _globals['_SCGETCARD']._serialized_start=3130
  _globals['_SCGETCARD']._serialized_end=3168
  _globals['_SCOPLIST']._serialized_start=3170
  _globals['_SCOPLIST']._serialized_end=3242
  _globals['_SCDZOPLIST']._serialized_start=3244
  _globals['_SCDZOPLIST']._serialized_end=3300
  _globals['_OPBUTTON']._serialized_start=3302
  _globals['_OPBUTTON']._serialized_end=3373
  _globals['_OPDZBUTTON']._serialized_start=3375
  _globals['_OPDZBUTTON']._serialized_end=3436
  _globals['_OPCONTEXT']._serialized_start=3439
  _globals['_OPCONTEXT']._serialized_end=3717
  _globals['_OPDZCONTEXT']._serialized_start=3720
  _globals['_OPDZCONTEXT']._serialized_end=4000
  _globals['_CHIOPCONTEXT']._serialized_start=4002
  _globals['_CHIOPCONTEXT']._serialized_end=4055
  _globals['_PENGOPCONTEXT']._serialized_start=4057
  _globals['_PENGOPCONTEXT']._serialized_end=4086
  _globals['_MINGGANGOPCONTEXT']._serialized_start=4088
  _globals['_MINGGANGOPCONTEXT']._serialized_end=4121
  _globals['_ANGANGOPCONTEXT']._serialized_start=4123
  _globals['_ANGANGOPCONTEXT']._serialized_end=4154
  _globals['_PENGGANGOPCONTEXT']._serialized_start=4156
  _globals['_PENGGANGOPCONTEXT']._serialized_end=4189
  _globals['_CSPASS']._serialized_start=4191
  _globals['_CSPASS']._serialized_end=4199
  _globals['_SCPASS']._serialized_start=4201
  _globals['_SCPASS']._serialized_end=4246
  _globals['_CSCHI']._serialized_start=4248
  _globals['_CSCHI']._serialized_end=4294
  _globals['_SCCHI']._serialized_start=4296
  _globals['_SCCHI']._serialized_end=4411
  _globals['_CSPENG']._serialized_start=4413
  _globals['_CSPENG']._serialized_end=4435
  _globals['_SCPENG']._serialized_start=4437
  _globals['_SCPENG']._serialized_end=4528
  _globals['_CSMINGGANG']._serialized_start=4530
  _globals['_CSMINGGANG']._serialized_end=4576
  _globals['_SCMINGGANG']._serialized_start=4578
  _globals['_SCMINGGANG']._serialized_end=4693
  _globals['_CSANGANG']._serialized_start=4695
  _globals['_CSANGANG']._serialized_end=4739
  _globals['_SCANGANG']._serialized_start=4741
  _globals['_SCANGANG']._serialized_end=4835
  _globals['_CSPENGGANG']._serialized_start=4837
  _globals['_CSPENGGANG']._serialized_end=4883
  _globals['_SCPENGGANG']._serialized_start=4885
  _globals['_SCPENGGANG']._serialized_end=4981
  _globals['_SCGETDICEGANGCARDS']._serialized_start=4983
  _globals['_SCGETDICEGANGCARDS']._serialized_end=5051
  _globals['_SCGIVEUPDICEGANGCARDS']._serialized_start=5053
  _globals['_SCGIVEUPDICEGANGCARDS']._serialized_end=5104
  _globals['_SCWAITOPWITHDICEGANGCARDS']._serialized_start=5106
  _globals['_SCWAITOPWITHDICEGANGCARDS']._serialized_end=5148
  _globals['_SCSTOPWAITOPWITHDICEGANGCARDS']._serialized_start=5150
  _globals['_SCSTOPWAITOPWITHDICEGANGCARDS']._serialized_end=5181
  _globals['_CSHU']._serialized_start=5183
  _globals['_CSHU']._serialized_end=5189
  _globals['_SCHU']._serialized_start=5192
  _globals['_SCHU']._serialized_end=5391
  _globals['_SCHU_PLAYERCARDS']._serialized_start=5332
  _globals['_SCHU_PLAYERCARDS']._serialized_end=5391
  _globals['_TINGPAIDATA']._serialized_start=5394
  _globals['_TINGPAIDATA']._serialized_end=5658
  _globals['_TINGPAIDATA_ENTRY']._serialized_start=5523
  _globals['_TINGPAIDATA_ENTRY']._serialized_end=5612
  _globals['_TINGPAIDATA_TINGCARDENTRY']._serialized_start=5614
  _globals['_TINGPAIDATA_TINGCARDENTRY']._serialized_end=5658
  _globals['_SCTINGPAIDATA']._serialized_start=5660
  _globals['_SCTINGPAIDATA']._serialized_end=5722
  _globals['_SCZHANIAO']._serialized_start=5724
  _globals['_SCZHANIAO']._serialized_end=5839
  _globals['_SCZHANIAO_ENTRY']._serialized_start=5784
  _globals['_SCZHANIAO_ENTRY']._serialized_end=5839
  _globals['_SCMANYOU']._serialized_start=5841
  _globals['_SCMANYOU']._serialized_end=5864
  _globals['_CSGETHAIDICARD']._serialized_start=5866
  _globals['_CSGETHAIDICARD']._serialized_end=5882
  _globals['_SCGETHAIDICARD']._serialized_start=5884
  _globals['_SCGETHAIDICARD']._serialized_end=5927
  _globals['_SCSTARTHU']._serialized_start=5929
  _globals['_SCSTARTHU']._serialized_end=6024
  _globals['_SCSTARTHU_ENTRY']._serialized_start=5989
  _globals['_SCSTARTHU_ENTRY']._serialized_end=6024
  _globals['_SCWAITOPWITHSTARTHU']._serialized_start=6026
  _globals['_SCWAITOPWITHSTARTHU']._serialized_end=6047
  _globals['_SCSTOPWAITOPWITHSTARTHU']._serialized_start=6049
  _globals['_SCSTOPWAITOPWITHSTARTHU']._serialized_end=6074
  _globals['_SCLIUJU']._serialized_start=6076
  _globals['_SCLIUJU']._serialized_end=6085
  _globals['_JIESUANSTARTHUBLOCK']._serialized_start=6088
  _globals['_JIESUANSTARTHUBLOCK']._serialized_end=6236
  _globals['_JIESUANSTARTHUBLOCK_HUENTRY']._serialized_start=6213
  _globals['_JIESUANSTARTHUBLOCK_HUENTRY']._serialized_end=6236
  _globals['_JIESUANHUBLOCK']._serialized_start=6238
  _globals['_JIESUANHUBLOCK']._serialized_end=6352
  _globals['_JIESUANPIAOBLOCK']._serialized_start=6354
  _globals['_JIESUANPIAOBLOCK']._serialized_end=6402
  _globals['_JIESUANZHANIAOBLOCK']._serialized_start=6404
  _globals['_JIESUANZHANIAOBLOCK']._serialized_end=6440
  _globals['_JIESUANPLAYER']._serialized_start=6443
  _globals['_JIESUANPLAYER']._serialized_end=6803
  _globals['_SCJIESUAN']._serialized_start=6806
  _globals['_SCJIESUAN']._serialized_end=6989
  _globals['_CSCONTINUENEXTROUND']._serialized_start=6991
  _globals['_CSCONTINUENEXTROUND']._serialized_end=7012
  _globals['_CSGM']._serialized_start=7014
  _globals['_CSGM']._serialized_end=7033
  _globals['_SCSTARTTIMER']._serialized_start=7035
  _globals['_SCSTARTTIMER']._serialized_end=7096
  _globals['_SCSTOPTIMER']._serialized_start=7098
  _globals['_SCSTOPTIMER']._serialized_end=7111
  _globals['_CSROUNDJIESUANVIEWCLOSED']._serialized_start=7113
  _globals['_CSROUNDJIESUANVIEWCLOSED']._serialized_end=7139
  _globals['_SCBLIND']._serialized_start=7141
  _globals['_SCBLIND']._serialized_end=7191
  _globals['_CSBELT']._serialized_start=7193
  _globals['_CSBELT']._serialized_end=7215
  _globals['_SCBELT']._serialized_start=7217
  _globals['_SCBELT']._serialized_end=7268
  _globals['_CSCALL']._serialized_start=7270
  _globals['_CSCALL']._serialized_end=7292
  _globals['_SCCALL']._serialized_start=7294
  _globals['_SCCALL']._serialized_end=7345
  _globals['_CSFILL']._serialized_start=7347
  _globals['_CSFILL']._serialized_end=7369
  _globals['_SCFILL']._serialized_start=7371
  _globals['_SCFILL']._serialized_end=7422
  _globals['_CSDISSCARD']._serialized_start=7424
  _globals['_CSDISSCARD']._serialized_end=7436
  _globals['_SCDISSCARD']._serialized_start=7438
  _globals['_SCDISSCARD']._serialized_end=7479
  _globals['_CSLOOKCARD']._serialized_start=7481
  _globals['_CSLOOKCARD']._serialized_end=7493
  _globals['_SCLOOKCARD']._serialized_start=7495
  _globals['_SCLOOKCARD']._serialized_end=7536
  _globals['_CSALLIN']._serialized_start=7538
  _globals['_CSALLIN']._serialized_end=7574
  _globals['_SCALLIN']._serialized_start=7576
  _globals['_SCALLIN']._serialized_end=7628
  _globals['_SCSTARTPREFLOP']._serialized_start=7630
  _globals['_SCSTARTPREFLOP']._serialized_end=7646
  _globals['_SCSTARTFLOP']._serialized_start=7648
  _globals['_SCSTARTFLOP']._serialized_end=7676
  _globals['_SCSTARTTURN']._serialized_start=7678
  _globals['_SCSTARTTURN']._serialized_end=7705
  _globals['_SCSTARTRIVER']._serialized_start=7707
  _globals['_SCSTARTRIVER']._serialized_end=7735
  _globals['_SCSTARTSHOWDOWN']._serialized_start=7737
  _globals['_SCSTARTSHOWDOWN']._serialized_end=7754
  _globals['_CSSITDOWN']._serialized_start=7756
  _globals['_CSSITDOWN']._serialized_end=7800
  _globals['_SCSITDOWN']._serialized_start=7802
  _globals['_SCSITDOWN']._serialized_end=7864
  _globals['_CSHOLDDESK']._serialized_start=7866
  _globals['_CSHOLDDESK']._serialized_end=7898
  _globals['_SCHOLDDESK']._serialized_start=7900
  _globals['_SCHOLDDESK']._serialized_end=7979
  _globals['_CSSTANDUP']._serialized_start=7981
  _globals['_CSSTANDUP']._serialized_end=8012
  _globals['_SCSTANDUP']._serialized_start=8014
  _globals['_SCSTANDUP']._serialized_end=8058
  _globals['_CSADDCHIPS']._serialized_start=8060
  _globals['_CSADDCHIPS']._serialized_end=8112
  _globals['_CSCANCELCHIPS']._serialized_start=8114
  _globals['_CSCANCELCHIPS']._serialized_end=8129
  _globals['_CSPLUSSCHIPS']._serialized_start=8131
  _globals['_CSPLUSSCHIPS']._serialized_end=8145
  _globals['_SCADDCHIPS']._serialized_start=8147
  _globals['_SCADDCHIPS']._serialized_end=8237
  _globals['_CSBACKGAME']._serialized_start=8239
  _globals['_CSBACKGAME']._serialized_end=8251
  _globals['_SCBACKGAME']._serialized_start=8253
  _globals['_SCBACKGAME']._serialized_end=8278
  _globals['_CSCANCELHOLDDESK']._serialized_start=8280
  _globals['_CSCANCELHOLDDESK']._serialized_end=8311
  _globals['_SCCANCELHOLDDESK']._serialized_start=8313
  _globals['_SCCANCELHOLDDESK']._serialized_end=8344
  _globals['_SCHOLDDESKSUC']._serialized_start=8346
  _globals['_SCHOLDDESKSUC']._serialized_end=8394
  _globals['_SCCANCLEDESKSUC']._serialized_start=8396
  _globals['_SCCANCLEDESKSUC']._serialized_end=8426
  _globals['_SCADDCHIPSSUC']._serialized_start=8428
  _globals['_SCADDCHIPSSUC']._serialized_end=8490
  _globals['_CSROOMPLAYERHANDRECORD']._serialized_start=8492
  _globals['_CSROOMPLAYERHANDRECORD']._serialized_end=8530
  _globals['_SCROOMPLAYERHANDRECORD']._serialized_start=8532
  _globals['_SCROOMPLAYERHANDRECORD']._serialized_end=8653
  _globals['_SCMACTHROOMBLINDINFO']._serialized_start=8655
  _globals['_SCMACTHROOMBLINDINFO']._serialized_end=8759
  _globals['_SCMACTHROOMISFINALTABLE']._serialized_start=8761
  _globals['_SCMACTHROOMISFINALTABLE']._serialized_end=8808
  _globals['_SCMATCHROOMREPEATINFO']._serialized_start=8810
  _globals['_SCMATCHROOMREPEATINFO']._serialized_end=8878
  _globals['_SCCLICKONTHEREPEAT']._serialized_start=8880
  _globals['_SCCLICKONTHEREPEAT']._serialized_end=8952
  _globals['_CSCLICKONTHEREPEAT']._serialized_start=8954
  _globals['_CSCLICKONTHEREPEAT']._serialized_end=8974
  _globals['_SCROOMPLAYEROPCOUNTDOWN']._serialized_start=8976
  _globals['_SCROOMPLAYEROPCOUNTDOWN']._serialized_end=9018
  _globals['_SCSTARTGAME']._serialized_start=9020
  _globals['_SCSTARTGAME']._serialized_end=9033
  _globals['_CSSHOWHANDCARDLISTINFO']._serialized_start=9035
  _globals['_CSSHOWHANDCARDLISTINFO']._serialized_end=9101
  _globals['_SHOWHANDCARD']._serialized_start=9103
  _globals['_SHOWHANDCARD']._serialized_end=9147
  _globals['_SCSETSHOWCARD']._serialized_start=9149
  _globals['_SCSETSHOWCARD']._serialized_end=9197
  _globals['_SCREQUESTHANDCARD']._serialized_start=9199
  _globals['_SCREQUESTHANDCARD']._serialized_end=9218
# @@protoc_insertion_point(module_scope)
