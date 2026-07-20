class TicketCodec:

    def checksum(self, ticket_id):
        # Position-weighted sum of character codes, mod 100.
        # Weighting by position (i+1) means changing ANY single character
        # -- or swapping two characters -- changes the total, not just an
        # unweighted sum which could miss reordering/certain edits.
        total = 0
        for i, char in enumerate(ticket_id):
            total += (i + 1) * ord(char)
        return total % 100

    def encode(self, ticket_id):
        checksum_value = self.checksum(ticket_id)
        return f"{ticket_id}-{checksum_value:02d}"

    def decode(self, encoded_ticket):
        try:
            ticket_id, checksum_str = encoded_ticket.rsplit('-', 1)
            checksum_value = int(checksum_str)
        except ValueError:
            return "CORRUPTED TICKET"

        if self.checksum(ticket_id) != checksum_value:
            return "CORRUPTED TICKET"
        return ticket_id

    def main(self):
        sample_ids = ["MIA2026GATE7", "MIA2026GATE12", "TRN27ALPHA"]
        encoded_tickets = []

        for ticket_id in sample_ids:
            encoded_ticket = self.encode(ticket_id)
            print(f"Encoded: {ticket_id} -> {encoded_ticket}")
            encoded_tickets.append(encoded_ticket)

        for encoded_ticket in encoded_tickets:
            decoded = self.decode(encoded_ticket)
            print(f"Decoded: {encoded_ticket} -> {decoded}")

   
        original = encoded_tickets[0]
        # flip the last character of the ticket ID portion (not the checksum)
        ticket_part, checksum_part = original.rsplit('-', 1)
        corrupted_char = 'X' if ticket_part[-1] != 'X' else 'Y'
        corrupted = ticket_part[:-1] + corrupted_char + '-' + checksum_part
        print(f"Corrupted: {original} -> {corrupted}")
        print(f"Decoded corrupted: {corrupted} -> {self.decode(corrupted)}")


if __name__ == "__main__":
    TicketCodec().main()