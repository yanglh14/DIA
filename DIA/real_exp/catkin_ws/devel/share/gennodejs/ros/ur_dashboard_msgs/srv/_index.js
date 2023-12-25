
"use strict";

let GetLoadedProgram = require('./GetLoadedProgram.js')
let IsProgramRunning = require('./IsProgramRunning.js')
let AddToLog = require('./AddToLog.js')
let GetProgramState = require('./GetProgramState.js')
let GetSafetyMode = require('./GetSafetyMode.js')
let IsInRemoteControl = require('./IsInRemoteControl.js')
let Load = require('./Load.js')
let Popup = require('./Popup.js')
let GetRobotMode = require('./GetRobotMode.js')
let RawRequest = require('./RawRequest.js')
let IsProgramSaved = require('./IsProgramSaved.js')

module.exports = {
  GetLoadedProgram: GetLoadedProgram,
  IsProgramRunning: IsProgramRunning,
  AddToLog: AddToLog,
  GetProgramState: GetProgramState,
  GetSafetyMode: GetSafetyMode,
  IsInRemoteControl: IsInRemoteControl,
  Load: Load,
  Popup: Popup,
  GetRobotMode: GetRobotMode,
  RawRequest: RawRequest,
  IsProgramSaved: IsProgramSaved,
};
