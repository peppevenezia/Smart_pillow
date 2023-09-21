/* ========================================
 *
 * Copyright YOUR COMPANY, THE YEAR
 * All Rights Reserved
 * UNPUBLISHED, LICENSED SOFTWARE.
 *
 * CONFIDENTIAL AND PROPRIETARY INFORMATION
 * WHICH IS THE PROPERTY OF your company.
 *
 * ========================================
*/
#include "project.h"
#include "InterruptRoutines.h"
uint8 ch_received,flag_connected=0;
extern uint8 flag_lettura;
uint8 count=0; 
extern float gain;
extern uint8 flag_calibr,flag_EOCalibr;
uint8 flag_count=1; 
 //after looking at the signal decide weather to do LP filter (start by sampling more frequently and avg for each sensor)
CY_ISR(Custom_ISR_RX){
    ch_received=UART_BT_GetChar();
    switch (ch_received){
        case 'b': //begin sampling and streaming
        case 'B':
            if (flag_connected==1){
                Timer_ADC_Start();
                PWM_LED_WritePeriod(32); //fast blinking (sampling and streaming)
                PWM_LED_WriteCompare(16);
                flag_calibr=1;
                gain=1;
            }
            else {
                PWM_LED_WritePeriod(255); //in case start is requested before connection (LED blinks slowly)
                PWM_LED_WriteCompare(128);   
            }
            
            break;
        
        case 's': //stop sampling and streaming
        case 'S':
            Timer_ADC_Stop();
            if (flag_connected==0){
                PWM_LED_WritePeriod(255); //in case stop is requested before connection (LED blinks slowly)
                PWM_LED_WriteCompare(128);   
            }
            else {
                PWM_LED_WritePeriod(255); //LED ON (connection is still established)
                PWM_LED_WriteCompare(255);
            }    
            break;
        
        case 'v': //search for correct port
            UART_BT_PutString("Posture$$$"); //GUI tries sending v to all connected ports and initializes communication only with the one which replies like this
            break;

        case 'e': //GUI sends e once it receives Posture$$$ and is ready to receive data
            PWM_LED_WritePeriod(255); //LED ON (conection established)
            PWM_LED_WriteCompare(255);
            flag_connected=1;
            break;  

    }

}


CY_ISR(Custom_ISR_ADC){
    Timer_ADC_ReadStatusRegister();

    flag_lettura=1; //at each ISR send signal to sample all sensors (done outside ISR in order not to block it)
    
    //flag_count=1; 
    if (flag_calibr==1) { //for calibration time window
        flag_count=1;
    }
    if(count==15) {//15 for 4,5 s or 33 9.9 seconds with ISR period of 300ms
        count=0;
        flag_calibr=0;
        flag_EOCalibr=1;
    }
}
/* [] END OF FILE */
