class SoundHelper():

    def __init__(self, enabled=True):
        self.playsound = None

        # Only import playsound if required
        if enabled:
            import playsound
            self.playsound = playsound

    def play_sound(self, sound_path) -> bool:
        if self.playsound is None:
            return False

        try:
            self.playsound.playsound(sound_path)
            return True
        except Exception as e: # Prevent the script from closing if the sound cannot be played
            print(f"Error playing sound {sound_path}: {e}")
            return False