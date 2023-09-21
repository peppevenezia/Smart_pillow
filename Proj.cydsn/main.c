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
#define Nsensors 11
#define BYTES_TO_SEND 2*Nsensors
#define TRANSMIT_BUFFER_SIZE 1+BYTES_TO_SEND+1
uint8 DataBuffer[TRANSMIT_BUFFER_SIZE]={0};
uint8 flag_lettura=0;
//uint8 flag_send=0;
int32 value,max=1;
float gain=1;
uint8 flag_calibr=1, flag_EOCalibr=0;

extern uint8 count, flag_count;
int main(void)
{
    CyGlobalIntEnable; /* Enable global interrupts. */

    PWM_LED_Start();
    PWM_LED_WritePeriod(255); //slow blinking (search for connection)
    PWM_LED_WriteCompare(128);
    
    AMux_Start();
    ADC_DelSig_Start();
    UART_BT_Start();
    isr_RX_StartEx(Custom_ISR_RX);
    isr_ADC_StartEx(Custom_ISR_ADC);
    
    //for (int j=1; j<=BYTES_TO_SEND;j++){
    //    DataBuffer[j]=0;    
    //}
    
    DataBuffer[0]=0xA0;
    DataBuffer[TRANSMIT_BUFFER_SIZE-1]=0xC0;
    
    for(;;)
    {
        
        if (flag_count==1 && flag_calibr==1) { //enters initially but continues only when the timers is started (by receiving "b")
            count++;
            flag_count=0;//to not count more times within the ISR
        }    
        
        if (flag_EOCalibr==1){
            gain=(65535.0)/max; //accounts for users having different weights, the sensors are always sensitive enough but the heatmap and the classification model should be invariant to user's weight offset so this is resolved already at level of passed data 
            flag_EOCalibr=0;
            //flag_calibr=0;
            max=1;
            PWM_LED_WritePeriod(64);  //change blinking to signal EOCalibration
            PWM_LED_WriteCompare(32);
            
            //flag_send=1;
        }
        
        
        if (flag_lettura ==1){
            for (int i=1; i<=Nsensors; i++){
                AMux_FastSelect(i-1);
                value=ADC_DelSig_Read32();
                if (gain<1) gain=1;
                value=value*gain;
                if (value>=65535) value=65535; //65534 
                if (value<0) value=0;
                if (value>max && flag_calibr==1) max=value;
                DataBuffer[2*i-1]= value >> 8; //MSB
                DataBuffer[2*i]= value & 0xff; //LSB
            }
            flag_lettura=0;
            //if (flag_send==1) { //to not send data during calibration
                UART_BT_PutArray(DataBuffer, TRANSMIT_BUFFER_SIZE);
            //}
        }
        
        
    }
}

/* [] END OF FILE */
