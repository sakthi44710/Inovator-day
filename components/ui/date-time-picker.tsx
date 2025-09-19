"use client"

import * as React from "react"
import { Calendar as CalendarIcon } from "lucide-react"
import { format } from "date-fns"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Calendar } from "./calendar"


const availableDates = [
  new Date(2024, 7, 25),
  new Date(2024, 7, 26),
  new Date(2024, 7, 27),
  new Date(2024, 7, 28),
  new Date(2024, 7, 29),
]

function DatePickerDemo({ setDate, date }: { setDate: (date: Date) => void, date: Date | undefined }) {
    return (
        <Calendar
            mode="single"
            selected={date}
            onSelect={(day) => day && setDate(day)}
            initialFocus
            disabled={(date) => !availableDates.find(d => d.toDateString() === date.toDateString())}
      />
    )
  }
  

export function DateTimePicker({
  date,
  setDate,
}: {
  date: Date | undefined
  setDate: (date: Date | undefined) => void
}) {
  const [selectedDate, setSelectedDate] = React.useState<Date | undefined>(date)
  const popoverRef = React.useRef<HTMLButtonElement>(null);


  React.useEffect(() => {
    setSelectedDate(date)
  }, [date])

  const handleDateChange = (newDate: Date | undefined) => {
    if(newDate){
        setSelectedDate(newDate)
        setDate(newDate)
        // Close popover on date select
        if (popoverRef.current) {
            popoverRef.current.click();
        }
    }
  }

  return (
    <Popover>
      <PopoverTrigger asChild ref={popoverRef}>
        <Button
          variant={"outline"}
          className={cn(
            "w-full justify-start text-left font-normal",
            !date && "text-muted-foreground"
          )}
        >
          <CalendarIcon className="mr-2 h-4 w-4" />
          {date ? format(date, "MMMM d, yyyy") : <span>Pick a date</span>}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-auto p-0">
        <DatePickerDemo 
          setDate={handleDateChange}
          date={selectedDate}
        />
      </PopoverContent>
    </Popover>
  )
}
